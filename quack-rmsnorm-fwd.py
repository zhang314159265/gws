import torch
from torch import Tensor
import operator
import math
import torch.nn.functional as F
from typing import Type, Callable, Optional
import triton
import cutlass
from cutlass import cute
from cutlass import Float16, BFloat16, Float32, Int32, Int64, Boolean
from cutlass import dsl_user_op
import cuda.bindings.driver as cuda
import os

M, N = int(os.getenv("SIZE_M", "32768")), int(os.getenv("SIZE_N", "4096"))

eps = 1e-5
x = torch.randn(M, N, device="cuda", dtype=torch.bfloat16)
w = torch.randn(N, device="cuda", dtype=torch.float32)
ref = F.rms_norm(x.float(), [N], w, eps).to(x.dtype)

def bench():
    act = quack_rmsnorm(x, w, eps)
    torch.testing.assert_close(ref, act)
    
    print("PASS!")
    
    ms = triton.testing.do_bench(lambda: quack_rmsnorm(x, w, eps))
    nbytes = x.nbytes * 2 + w.nbytes
    gbps = (nbytes * 1e-9) / (ms * 1e-3)
    print(f"{ms:.3f} ms, {gbps:.3f} gbps")

def quack_rmsnorm(x, w, eps):
    out = torch.empty_like(x)
    _rmsnorm_fwd(x, w, out, eps)
    return out

torch2cute_dtype_map = {
    torch.bfloat16: BFloat16,
    torch.float16: Float16,
    torch.float32: Float32,
    torch.int32: Int32,
    torch.int64: Int64,
}

def layout_utils_expand(tensor: cute.Tensor, dim: int, size: int | Int32) -> cute.Tensor:
    new_shape = (*tensor.shape[:dim], size, *tensor.shape[dim:])
    new_stride = (*tensor.stride[:dim], 0, *tensor.stride[dim:])
    return cute.make_tensor(tensor.iterator, cute.make_layout(new_shape, stride=new_stride))

def get_vecsize(*tensors: cute.Tensor, N: int):
    max_width = cutlass.const_expr(max(t.element_type.width for t in tensors))
    vecsize = 128 // max_width
    vecsize = math.gcd(N, vecsize)
    print(f"{vecsize=}")
    return vecsize

@dsl_user_op
def get_copy_atom(dtype: Type[cutlass.Numeric], num_copy_elems: int, *, loc=None, ip=None) -> cute.CopyAtom:
    copy_op = cute.nvgpu.CopyUniversalOp()
    num_bits_per_copy = cutlass.const_expr(num_copy_elems * dtype.width)
    return cute.make_copy_atom(copy_op, dtype, num_bits_per_copy=num_bits_per_copy, loc=loc, ip=ip)

def ceil_div(a, b):
    return (a + b - 1) // b


def allocate_reduction_buffer(smem: cutlass.utils.SmemAllocator, reduction_dtype: Type[cutlass.Numeric], tv_layout: cute.Layout):
    threads_per_row, num_row = tv_layout.shape[0]
    warps_per_row = threads_per_row // cute.arch.WARP_SIZE
    buf_layout = cute.make_ordered_layout(
        (num_row, warps_per_row), order=(1, 0)
    )
    return smem.allocate_tensor(reduction_dtype, buf_layout, byte_alignment=8)

@cute.jit # why this @cute.jit annotation is required
def block_reduce(
    val: cute.Numeric, op: Callable, reduction_buffer: cute.Tensor, zero_val: cute.Numeric,
) -> cute.Numeric:
    """
    val is already the warp level reduction result. Each warp contains the
    same value.
    """
    lane_id = cute.arch.lane_idx()
    warp_id = cute.arch.warp_idx()

    warps_per_row = reduction_buffer.shape[1]
    row_id = warp_id // warps_per_row
    warp_id_in_row = warp_id % warps_per_row

    if lane_id == 0:
        # lane 0 of each warp write 'val' to smem
        reduction_buffer[row_id, warp_id_in_row] = val

    cute.arch.barrier()
    val = zero_val
    if lane_id < warps_per_row:
        val = reduction_buffer[row_id, lane_id]
    return cute.arch.warp_reduction(
        val,
        op,
    )

# @cute.jit # why this @cute.jit annotation is optional
def row_reduce(
    x: cute.TensorSSA,
    reduction_buffer: cute.Tensor,
    zero_val: cute.Numeric,
) -> cute.Numeric:
    val = x.reduce(cute.ReductionOp.ADD, zero_val, 0)
    combine_op = operator.add
    warp_val = cute.arch.warp_reduction(val, combine_op)
    if cutlass.const_expr(reduction_buffer.shape[1] > 1):
        return block_reduce(warp_val, combine_op, reduction_buffer, zero_val)
    else:
        return warp_val

@dsl_user_op
def copy_util(
    src: cute.Tensor,
    dst: cute.Tensor,
    *,
    loc=None,
    ip=None,
):
    vecsize = src.shape[0][0]
    atom = get_copy_atom(src.element_type, vecsize, loc=loc, ip=ip)
    cute.copy(atom, src, dst)

class RMSNorm:
    def __init__(self, dtype: Type[cutlass.Numeric], N: int):
        self.dtype = dtype
        self.N = N
        self.reduction_dtype = Float32

    @cute.jit
    def __call__(
        self,
        mX: cute.Tensor,
        mW: cute.Tensor,
        mO: cute.Tensor,
        eps: Float32,
        stream: cuda.CUstream,
    ):
        vecsize = get_vecsize(mX, mW, mO, N=self.N)
        tiled_copy, tiler_mn, threads_per_row = self._get_tiled_copy(vecsize)
        mW = layout_utils_expand(mW, 0, tiler_mn[0])
        num_threads = self._num_threads()
        assert num_threads == tiled_copy.size
    
        self.kernel(mX, mW, mO, eps, tiler_mn, tiled_copy, threads_per_row).launch(
            block=(num_threads, 1, 1),
            grid=(ceil_div(mX.shape[0], tiler_mn[0]), 1, 1),
        )

    def _get_tiled_copy(self, vecsize):
        """
        Returns
            (TiledCopy, tiler_mn, threads_per_row)
        """
        num_threads = self._num_threads()
        threads_per_row = self._threads_per_row()
        num_row = num_threads // threads_per_row
    
        thread_layout = cute.make_ordered_layout((num_row, threads_per_row), order=(1, 0))
        value_layout = cute.make_layout((1, vecsize))
        atom = get_copy_atom(self.dtype, vecsize)
        tiled_copy = cute.make_tiled_copy_tv(atom, thread_layout, value_layout)
    
        ntile_per_thread = ceil_div(self.N, threads_per_row * vecsize)
        tiler_mn = (
            num_row,
            ntile_per_thread * vecsize * threads_per_row,
        )
        return tiled_copy, tiler_mn, threads_per_row

    @cute.kernel
    def kernel(
        self,
        mX: cute.Tensor,
        mW: cute.Tensor,
        mO: cute.Tensor,
        eps: Float32,
        tiler_mn: cute.Shape,
        tiled_copy: cute.TiledCopy,
        threads_per_row: cutlass.Constexpr[int],
    ):
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
    
        smem = cutlass.utils.SmemAllocator()
    
        tv_layout = tiled_copy.layout_tv_tiled
    
        reduction_buffer = allocate_reduction_buffer(smem, self.reduction_dtype, tv_layout)
    
        gX = cute.local_tile(mX, tiler_mn, (bidx, 0))
        gO = cute.local_tile(mO, tiler_mn, (bidx, 0))
        gW = cute.local_tile(mW, tiler_mn, (0, 0))
    
        thr_copy = tiled_copy.get_slice(tidx)
        tXgX = thr_copy.partition_S(gX)
        tXgW = thr_copy.partition_S(gW)
        tXgO = thr_copy.partition_D(gO)
    
        tXrX = cute.make_fragment_like(tXgX)
        tXrW = cute.make_fragment_like(tXgW)
        tXrO = cute.make_fragment_like(tXgO)
        copy_util(tXgX, tXrX)
        copy_util(tXgW, tXrW)
    
        x = tXrX.load().to(Float32)
        y = row_reduce(x * x, reduction_buffer, 0.0)
        y = cute.math.rsqrt(y / self.N + eps, fastmath=True)
    
        w = tXrW.load()
        y = x * y * w
        tXrO.store(y.to(BFloat16))
        copy_util(tXrO, tXgO)
    
    def _num_threads(self):
        return 128 if self.N <= 16384 else 256
    
    # TODO test for threads_per_row < 32
    def _threads_per_row(self):
        for limit, threads in [(64, 8), (128, 16), (3072, 32), (6144, 64), (16384, 128)]:
            if self.N <= limit:
                return threads
    
        return 256

def make_fake_tensor(dtype, shape, divisibility=1) -> cute.Tensor:
    stride = [
        1
        if i == len(shape) - 1
        else
        cute.sym_int(divisibility=divisibility)
        for i in range(len(shape))
    ]
    return cute.runtime.make_fake_tensor(dtype, shape, stride=stride, assumed_align=divisibility * dtype.width // 8)

def _rmsnorm_fwd(
    x: Tensor,
    weight: Tensor,
    out: Tensor,
    eps: float
) -> None:
    xdtype, wdtype, odtype = dtypes = [torch2cute_dtype_map[t.dtype] for t in (x, weight, out)]
    _, N = x.shape
    compile_key = (*dtypes, N)
    if compile_key not in _rmsnorm_fwd.compile_cache:
        vecsize = math.gcd(N, *(128 // t.width for t in dtypes))
        bs = cute.sym_int()
        x_cute = make_fake_tensor(xdtype, (bs, N), divisibility=vecsize)
        out_cute = make_fake_tensor(odtype, (bs, N), divisibility=vecsize)
        w_cute = make_fake_tensor(wdtype, (N,), divisibility=vecsize)

        _rmsnorm_fwd.compile_cache[compile_key] = cute.compile(
            RMSNorm(xdtype, N),
            x_cute,
            w_cute,
            out_cute,
            Float32(0), # eps, just for compilation
            cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),
            options="--enable-tvm-ffi",
        )

    _rmsnorm_fwd.compile_cache[compile_key](
        x, weight, out, eps
    )

_rmsnorm_fwd.compile_cache = {}

bench()
