
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.runtime.triton_heuristics import AutotuneHint, reduction
from torch._inductor.runtime import triton_helpers

from torch._dynamo.testing import rand_strided
from torch._C import _cuda_getCurrentRawStream as get_cuda_stream
import torch
from torch._inductor.runtime.triton_heuristics import grid
from triton.compiler.compiler import AttrsDescriptor
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties

@reduction(
    size_hints={'x': 8192, 'r0_': 65536},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {"in_ptr0": '*fp32', "out_ptr2": '*fp32', "xnumel": 'i32', "rnumel": 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=108, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused__softmax_0', 'mutated_arg_names': [], "kernel_num_gb": 0}
)
@triton.jit
def triton_red_fused__softmax_0(in_ptr0, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    xnumel = 8192
    rnumel = 65536
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, R0_BLOCK)[None, :]
    x0 = xindex
    _tmp2 = tl.full([XBLOCK, R0_BLOCK], float("-inf"), tl.float32)
    for roffset in range(0, rnumel, R0_BLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp0 = tl.load(in_ptr0 + (r1 + (65536*x0)), rmask, eviction_policy='evict_last', other=0.0)
        tmp1 = tl.broadcast_to(tmp0, [XBLOCK, R0_BLOCK])
        tmp3 = triton_helpers.maximum(_tmp2, tmp1)
        _tmp2 = tl.where(rmask, tmp3, _tmp2)
    tmp2 = triton_helpers.max2(_tmp2, 1)[:, None]
    _tmp8 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, R0_BLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp4 = tl.load(in_ptr0 + (r1 + (65536*x0)), rmask, eviction_policy='evict_last', other=0.0)
        tmp5 = tmp4 - tmp2
        tmp6 = tl.exp(tmp5)
        tmp7 = tl.broadcast_to(tmp6, [XBLOCK, R0_BLOCK])
        tmp9 = _tmp8 + tmp7
        _tmp8 = tl.where(rmask, tmp9, _tmp8)
    tmp8 = tl.sum(_tmp8, 1)[:, None]
    for roffset in range(0, rnumel, R0_BLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp10 = tl.load(in_ptr0 + (r1 + (65536*x0)), rmask, eviction_policy='evict_first', other=0.0)
        tmp11 = tmp10 - tmp2
        tmp12 = tl.exp(tmp11)
        tmp13 = tmp12 / tmp8
        tl.store(out_ptr2 + (r1 + (65536*x0)), tmp13, rmask)


def get_args():
    arg_0 = rand_strided((8192, 65536), (65536, 1), device='cuda:0', dtype=torch.float32)
    arg_1 = rand_strided((8192, 65536), (65536, 1), device='cuda:0', dtype=torch.float32)
    return arg_0, arg_1,


def call(args):
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        stream0 = get_cuda_stream(0)
        triton_red_fused__softmax_0.run(*args, 8192, 65536, grid=grid(8192), stream=stream0)


def benchmark_all_configs(args):
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        return triton_red_fused__softmax_0.benchmark_all_configs(*args, 8192, 65536, grid=grid(8192))


if __name__ == '__main__':
    from torch._inductor.runtime.benchmarking import benchmarker

    args = get_args()
    ms = benchmarker.benchmark_gpu(lambda: call(args), rep=40)
    num_gb = 0.0
    gb_per_s = num_gb / (ms / 1e3)
    print(f"{ms:.3f}ms    {num_gb:.3f}GB    {gb_per_s:.2f}GB/s")
