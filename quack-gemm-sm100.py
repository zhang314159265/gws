import time
from enum import IntEnum
import enum
import os
from triton.testing import do_bench
import torch
import argparse

import cutlass
from cutlass import cute
from cutlass.base_dsl.typing import DynamicExpression, JitArgument, NumericMeta, get_c_pointers, get_mlir_types
from cutlass.cute.runtime import from_dlpack
import cutlass.torch
from cutlass import Int32, Uint32, Boolean, const_expr, Constexpr
from cutlass.utils.layout import LayoutEnum
import cutlass.utils.blackwell_helpers as sm100_utils

from typing import Type, Callable, Literal
from dataclasses import dataclass, fields
from functools import partial

torch.manual_seed(41)

def parse_comma_separated_ints(s: str) -> tuple[int, ...]:
    return tuple(int(part.strip()) for part in s.split(","))

def main():
    parser = argparse.ArgumentParser("Dense Persistent Gemm for Blackwell")
    parser.add_argument(
        "--mnkl",
        type=parse_comma_separated_ints,
        default=(4096, 4096, 4096, 2) if os.getenv("LARGE_INPUT") == "1" else (256, 256, 512, 2),
        help="mnkl dimensions",
    )
    parser.add_argument(
        "--mma_tiler_mn",
        type=parse_comma_separated_ints,
        default=(128, 128),
        help="Mma tile shape",
    )
    parser.add_argument(
        "--cluster_shape_mn",
        type=parse_comma_separated_ints,
        default=(1, 1),
        help="cluster shape",
    )
    parser.add_argument("--ab_dtype", type=cutlass.dtype, default=cutlass.BFloat16)
    parser.add_argument("--d_dtype", type=cutlass.dtype, default=cutlass.BFloat16)
    parser.add_argument("--acc_dtype", type=cutlass.dtype, default=cutlass.Float32)

    args = parser.parse_args()
    run(
        args.mnkl,
        args.mma_tiler_mn,
        args.cluster_shape_mn,
        args.ab_dtype,
        args.d_dtype,
        args.acc_dtype,
    )
    print("PASS")

StaticTypes = (cutlass.Constexpr, NumericMeta, int, float, bool, str, type(None))

class MLIRSerdeBase:
    def _extract_mlir(self, fn):
        """
        Extract mlir values from object.
        """
        nitem_per_field = []
        values = []
        for f in fields(self):
            name = f.name 
            val = getattr(self, name)
    
            if not isinstance(val, StaticTypes):
                new_values = fn(val)
                values.extend(new_values)
                nitem_per_field.append(len(new_values))
    
        if not hasattr(self, "_nitem_per_field"):
            self._nitem_per_field = nitem_per_field
        else:
            assert self._nitem_per_field == nitem_per_field
        return values

    def __new_from_mlir_values__(self, values):
        """
        Construct object from mlir values.
        """
        const_val_dict = {}
        nonconst_val_dict = {}
        get_nitem = iter(self._nitem_per_field)

        for f in fields(self):
            name = f.name
            val = getattr(self, name)

            if isinstance(val, StaticTypes):
                const_val_dict[name] = val
            else:
                nitem = next(get_nitem)
                nonconst_val_dict[name] = cutlass.new_from_mlir_values(
                    val, values[:nitem]
                )
                values = values[nitem:]

        assert len(list(get_nitem)) == 0, "Unused items found in self._nitem_per_field"
        return self.__class__(**const_val_dict, **nonconst_val_dict)

@dataclass
class ParamsBase(MLIRSerdeBase, DynamicExpression):
    def __extract_mlir_values__(self):
        return self._extract_mlir(fn=cutlass.extract_mlir_values)

@dataclass
class ArgumentsBase(MLIRSerdeBase, JitArgument):
    def __c_pointers__(self):
        return self._extract_mlir(fn=get_c_pointers)

    def __get_mlir_types__(self):
        return self._extract_mlir(fn=get_mlir_types)

def create_and_permute_tensor(l, mode0, mode1, cute_dtype):
    torch_dtype = cutlass.torch.dtype(cute_dtype)
    torch_tensor = cutlass.torch.create_and_permute_torch_tensor(
        [l, mode0, mode1],
        dtype=torch_dtype,
        permute_order=[1, 2, 0],
        init_type=cutlass.torch.TensorInitType.GAUSSIAN,
        device="cuda"
    )
    cute_tensor = from_dlpack(torch_tensor, assumed_align=16)
    cute_tensor = cute_tensor.mark_layout_dynamic()
    return torch_tensor, cute_tensor

def run(
    mnkl: tuple[int, int, int, int],
    mma_tiler_mn: tuple[int, int],
    cluster_shape_mn: tuple[int, int],
    ab_dtype: Type[cutlass.Numeric],
    d_dtype: Type[cutlass.Numeric],
    acc_dtype: Type[cutlass.Numeric],
    tolerance: float = 1e-2,
):
    print("Running Blackwell Persistent Dense GEMM test with:")
    print(f"AB dtype: {ab_dtype}, D dtype: {d_dtype}, Acc dtype: {acc_dtype}")
    print(f"Mma tiler (M, N): {mma_tiler_mn}, cluster shape (M, N): {cluster_shape_mn}")
    print(f"Tolerance: {tolerance}")

    m, n, k, l = mnkl
    torch_a, cute_a = create_and_permute_tensor(l, m, k, ab_dtype)
    torch_b, cute_b = create_and_permute_tensor(l, n, k, ab_dtype)
    torch_d, cute_d = create_and_permute_tensor(l, m, n, d_dtype)

    cluster_shape_mnk = (*cluster_shape_mn, 1)
    gemm = GemmSm100(acc_dtype, mma_tiler_mn, cluster_shape_mnk)

    hardware_info = cutlass.utils.HardwareInfo()
    max_active_clusters = hardware_info.get_max_active_clusters(1)

    torch_stream = torch.cuda.current_stream()
    current_stream = cuda.CUstream(torch_stream.cuda_stream)

    compiled_gemm = cute.compile(
        gemm,
        cute_a,
        cute_b,
        cute_d,
        max_active_clusters,
        current_stream,
    )

    compiled_gemm(cute_a, cute_b, cute_d, max_active_clusters, current_stream)
    ref = torch.einsum("mkl,nkl->mnl", torch_a.float(), torch_b.float()).to(cutlass.torch.dtype(d_dtype))
    torch.testing.assert_close(ref, torch_d, atol=tolerance, rtol=tolerance)

    # perf benchmark
    flops = 2 * m * n * k * l
    fn_cublas = lambda: torch.matmul(torch_a.permute(2, 0, 1), torch_b.permute(2, 1, 0))
    timing_cublas = do_bench(fn_cublas)
    tflops_cublas = (flops * 1e-12) / (timing_cublas * 1e-3)
    print(f"CuBLAS average time: {timing_cublas:.3f} ms, TFLOPS: {tflops_cublas:.1f}")

    fn_cute = lambda: compiled_gemm(
        cute_a, cute_b, cute_d, max_active_clusters, current_stream
    )
    timing_cute = do_bench(fn_cute)
    tflops_cute = (flops * 1e-12) / (timing_cute * 1e-3)
    print(f"CuTeDSL average time: {timing_cute:.3f} ms, TFLOPS: {tflops_cute:.1f}")

class RasterOrder(IntEnum):
    AlongM = 0
    AlongN = 1

def GemmSm100_get_scheduler_params(
    self,
    mA: cute.Tensor,
    mB: cute.Tensor,
    mD: cute.Tensor,
):
    problem_shape_ntile_mnl = (
        cute.ceil_div(mA.shape[0], self.cta_tile_shape_mnk[0]),
        cute.ceil_div(mB.shape[0], self.cta_tile_shape_mnk[1]),
        mA.shape[2],
    )
    args = TileSchedulerArguments(
        problem_shape_ntile_mnl,
        group_size=8,
        cluster_shape_mnk=self.cluster_shape_mnk,
    )
    return TileSchedulerParams.create(args)

@dataclass
class TileSchedulerArguments:
    problem_shape_ntile_mnl: cute.Shape
    group_size: int
    cluster_shape_mnk: cutlass.Constexpr[cute.Shape]

@cute.jit
def get_raster_order(
    problem_shape_ncluster_mn: cute.Shape, group_size: int,
) -> RasterOrder:
    m = cute.round_up(problem_shape_ncluster_mn[0], group_size)
    n = cute.round_up(problem_shape_ncluster_mn[1], group_size)
    return RasterOrder.AlongM if m < n else RasterOrder.AlongN

def _divmod(a, b):
    return a // b, a % b

@dataclass
class TileSchedulerParams(ParamsBase):
    problem_shape_ncluster_mnl: cute.Shape
    cluster_shape_mn: cutlass.Constexpr[cute.Shape]
    num_clusters_per_problem: Int32
    num_clusters_in_group: Int32
    num_groups_regular: Int32
    group_size_tail: Int32
    group_size: Int32
    raster_order: RasterOrder

    @staticmethod
    @cute.jit
    def create(args: TileSchedulerArguments) -> "TileSchedulerParams":
        problem_shape_ntile_mnl = args.problem_shape_ntile_mnl
        group_size = args.group_size
        cluster_shape_mnk = args.cluster_shape_mnk

        cluster_shape_mn = cutlass.const_expr(cute.select(cluster_shape_mnk, mode=(0, 1)))
        problem_shape_ntile_mn = cute.select(problem_shape_ntile_mnl, mode=(0, 1))
        problem_shape_ncluster_mn = cute.ceil_div(problem_shape_ntile_mn, cluster_shape_mn)
        problem_shape_ncluster_mnl = (*problem_shape_ncluster_mn, problem_shape_ntile_mnl[2])
        num_clusters_per_problem = cute.size(problem_shape_ncluster_mn)

        raster_order = get_raster_order(
            problem_shape_ncluster_mn, group_size
        )

        ncluster_fast = problem_shape_ncluster_mn[0] if raster_order == RasterOrder.AlongM else problem_shape_ncluster_mn[1]
        ncluster_slow = problem_shape_ncluster_mn[1] if raster_order == RasterOrder.AlongM else problem_shape_ncluster_mn[0]
        group_size = min(group_size, ncluster_fast)

        num_clusters_in_group = group_size * ncluster_slow

        num_groups_regular, group_size_tail = _divmod(ncluster_fast, group_size)

        return TileSchedulerParams(
            problem_shape_ncluster_mnl=problem_shape_ncluster_mnl,
            cluster_shape_mn=cluster_shape_mn,
            num_clusters_per_problem=num_clusters_per_problem,
            num_clusters_in_group=num_clusters_in_group,
            num_groups_regular=num_groups_regular,
            group_size_tail=group_size_tail,
            group_size=group_size,
            raster_order=raster_order,
        )

@dataclass
class TileScheduler:
    _current_work_linear_idx: Int32
    params: TileSchedulerParams
    num_tiles_executed: int = 0

    @staticmethod
    def get_grid_shape(
        params: TileSchedulerParams,
        max_active_clusters: int,
    ) -> tuple[Int32, Int32, Int32]:
        num_persistent_ctas = cutlass.min(
            cute.size(params.problem_shape_ncluster_mnl),
            max_active_clusters,
        )
        return (1, 1, num_persistent_ctas)
    
    def advance_to_next_work(
        self,
    ):
        self._current_work_linear_idx += cute.arch.grid_dim()[2]
        self.num_tiles_executed += 1
    
    @cute.jit
    def _swizzle_cta(
        self, cluster_id_in_problem: Int32,
    ) -> tuple[Int32, Int32]:
        group_id, id_in_group = _divmod(cluster_id_in_problem, self.params.num_clusters_in_group)
        cid_slow, cid_fast_in_group = 0, 0
        if group_id < self.params.num_groups_regular:
            cid_slow, cid_fast_in_group = _divmod(id_in_group, self.params.group_size) 
        else:
            cid_slow, cid_fast_in_group = _divmod(id_in_group, self.params.group_size_tail)
        cid_fast = cid_fast_in_group + group_id * self.params.group_size
        cid_m, cid_n = 0, 0
        if self.params.raster_order == RasterOrder.AlongM:
            cid_m, cid_n = cid_fast, cid_slow
        else:
            cid_m, cid_n = cid_slow, cid_fast
        return cid_m, cid_n
    
    def create(
        params: TileSchedulerParams,
    ) -> "TileScheduler":
        return TileScheduler(
            cute.arch.block_idx()[2],
            params,
        )
    
    def get_current_work(self) -> cutlass.utils.WorkTileInfo:
        bidz, cluster_id_in_problem = _divmod(self._current_work_linear_idx, self.params.num_clusters_per_problem)
        pid_m, pid_n = self._swizzle_cta(cluster_id_in_problem)
        return cutlass.utils.WorkTileInfo(
            (pid_m, pid_n, None, bidz), # mnkl
            self._current_work_linear_idx < cute.size(self.params.problem_shape_ncluster_mnl)
        )

def GemmSm100___init__(
    self,
    acc_dtype: Type[cutlass.Numeric],
    mma_tiler_mn: tuple[int, int],
    cluster_shape_mnk: tuple[int, int, int],
):
    self.acc_dtype = acc_dtype
    self.cluster_shape_mnk = cluster_shape_mnk
    self.mma_tiler = (*mma_tiler_mn, 1)
    self.cta_group = tcgen05.CtaGroup.ONE

    self.epilog_warp_id = (0, 1, 2, 3)
    self.mma_warp_id = 4
    self.ab_load_warp_id = 5
    self.threads_per_cta = cute.arch.WARP_SIZE * (self.num_epi_warps + 2)

    self.buffer_align_bytes = 1024

def copy_utils_cvt_copy(
    tiled_copy: cute.TiledCopy,
    src: cute.Tensor,
    dst: cute.Tensor,
):
    # src must be in register
    assert isinstance(src.iterator, cute.Pointer) and src.memspace == cute.AddressSpace.rmem
    if const_expr(src.element_type != dst.element_type):
        src_cvt = cute.make_fragment_like(src, dst.element_type)
        src_cvt.store(src.load().to(dst.element_type))
        src = src_cvt
    cute.copy(tiled_copy, src, dst)

class NamedBarrierEnum(enum.IntEnum):
    # start from 1
    TmemPtr = enum.auto()
    Epilogue = enum.auto()

# ============ XXX SPLIT ===========

from cutlass.cutlass_dsl import if_generate
from cutlass.pipeline import PipelineAsync, PipelineTmaAsync, PipelineState, PipelineUserType

import cuda.bindings.driver as cuda

from cutlass.cute.nvgpu import cpasync, tcgen05
from cutlass.cutlass_dsl import T, dsl_user_op

def copy_utils_tma_get_copy_fn(
    atom: cute.CopyAtom,
    cta_coord: cute.Coord,
    cta_layout: cute.Layout,
    src_tensor: cute.Tensor,
    dst_tensor: cute.Tensor,
    **kwargs,
) -> Callable:
    src_is_smem = const_expr(
        isinstance(src_tensor.iterator, cute.Pointer)
        and src_tensor.memspace == cute.AddressSpace.smem
    )
    smem_tensor, gmem_tensor = (src_tensor, dst_tensor) if src_is_smem else (dst_tensor, src_tensor)
    group_rank_smem = const_expr(cute.rank(smem_tensor) - 1)
    group_rank_gmem = const_expr(cute.rank(gmem_tensor) - 1)
    s, g = cpasync.tma_partition(
        atom,
        cta_coord,
        cta_layout,
        cute.group_modes(smem_tensor, 0, group_rank_smem),
        cute.group_modes(gmem_tensor, 0, group_rank_gmem),
    )
    src, dst = (s, g) if src_is_smem else (g, s)

    def copy_tma(src_idx, dst_idx, **new_kwargs):
        cute.copy(atom, src[None, src_idx], dst[None, dst_idx], **new_kwargs, **kwargs)

    return copy_tma


class GemmSm100:
    get_scheduler_params = GemmSm100_get_scheduler_params
    __init__ = GemmSm100___init__

    def epilog_gmem_copy_and_partition(
        self,
        atom: cute.CopyAtom | cute.TiledCopy,
        mD_mn: cute.Tensor,
        tile_shape_mn: cute.Tile,
        epi_tile: cute.Tile,
        sD: cute.Tensor,
        tile_coord_mnkl: cute.Coord,
    ):
        # (bM, bN)
        gD = cute.local_tile(mD_mn, tile_shape_mn, tile_coord_mnkl[:2])
        tDgD_for_tma_partition = cute.zipped_divide(gD, epi_tile)
        is_s2g = isinstance(
            atom.op, (cpasync.CopyBulkTensorTileS2GOp, cpasync.CopyReduceBulkTensorTileS2GOp)
        )
        src_tensor, dst_tensor = (
            (sD, tDgD_for_tma_partition) if is_s2g else (tDgD_for_tma_partition, sD)
        )
        return copy_utils_tma_get_copy_fn(
            atom,
            cta_coord=0,
            cta_layout=cute.make_layout(1),
            src_tensor=src_tensor,
            dst_tensor=dst_tensor,
        )

    @cute.jit
    def load_AB(
        self,
        ab_pipeline: cutlass.pipeline.PipelineAsync,
        ab_producer_state: cutlass.pipeline.PipelineState,
        copy_A: Callable,
        copy_B: Callable,
        k_tile_cnt: Int32,
    ) -> cutlass.pipeline.PipelineState:
        # Peek (try_wait) AB buffer empty for k_block = prefetch_k_tile_cnt
        peek_ab_empty_status = Boolean(True)
        if 0 < k_tile_cnt:
            peek_ab_empty_status = ab_pipeline.producer_try_acquire(ab_producer_state)
        for k_tile in cutlass.range(k_tile_cnt, unroll=1):
            # Wait for A/B buffers to be empty before loading into them
            # Also sets the transaction barrier for the A/B buffers
            ab_pipeline.producer_acquire(ab_producer_state, peek_ab_empty_status)
            tma_bar_ptr = ab_pipeline.producer_get_barrier(ab_producer_state)
            smem_idx = ab_producer_state.index
            copy_A(k_tile, smem_idx, tma_bar_ptr=tma_bar_ptr)
            copy_B(k_tile, smem_idx, tma_bar_ptr=tma_bar_ptr)
            # Mainloop pipeline's producer commit is a NOP
            ab_pipeline.producer_commit(ab_producer_state)
            ab_producer_state.advance()
            peek_ab_empty_status = Boolean(True)
            if k_tile + 1 < k_tile_cnt:
                peek_ab_empty_status = ab_pipeline.producer_try_acquire(ab_producer_state)
        return ab_producer_state

    @staticmethod
    def _make_tma_epi_atoms_and_tensors(
        tensor_d: cute.Tensor,
        epi_smem_layout_staged: cute.ComposedLayout,
        epi_tile: tuple[int, int],
        op_type: Literal["store", "load", "add"],
    ) -> tuple[cute.CopyAtom, cute.Tensor]:
        assert op_type in ["load", "store", "add"]
        epi_smem_layout = cute.slice_(epi_smem_layout_staged, (None, None, 0))
        d_cta_v_layout = cute.composition(cute.make_identity_layout(tensor_d.shape), epi_tile)
        op = (
            cpasync.CopyBulkTensorTileG2SOp()
            if op_type == "load"
            else cpasync.CopyBulkTensorTileS2GOp()
            if op_type == "store"
            else cpasync.CopyReduceBulkTensorTileS2GOp(cute.ReductionOp.ADD)
        )
        tma_atom_d, tma_tensor_d = cpasync.make_tiled_tma_atom(
            op, tensor_d, epi_smem_layout, d_cta_v_layout
        )
        return tma_atom_d, tma_tensor_d

    @property
    def num_epi_warps(self):
        return len(self.epilog_warp_id)

    @cute.jit
    def epilogue(
        self,
        params: ParamsBase,
        epi_store_pipeline: cutlass.pipeline.PipelineAsync,
        epi_read_state: cutlass.pipeline.PipelineState,
        epi_producer_state: cutlass.pipeline.PipelineState | None,
        epi_tile: cute.Tile,
        load_acc_subtile: Callable,
        tRS_rD: cute.Tensor,
        tiled_copy_t2r: cute.TiledCopy | None,  # Only for Sm100
        tiled_copy_r2s: cute.TiledCopy,
        tRS_sD: cute.Tensor,
        copy_D: Callable,
        tile_coord_mnkl: cute.Coord,
        epilogue_barrier: cutlass.pipeline.NamedBarrier,
        tile_scheduler,
        tidx: Int32,
        is_tma_warp: Boolean,
    ) -> tuple[cutlass.pipeline.PipelineState, cutlass.pipeline.PipelineState]:
        epi_tile_shape = cute.zipped_divide(
            cute.make_layout(self.cta_tile_shape_mnk[:2]), epi_tile
        ).shape[1]
        # We iterate over epi tiles in the N dimension first before the M dimension
        epi_tile_layout = cute.make_ordered_layout(epi_tile_shape, order=(1, 0))
        epi_tile_num = cute.size(epi_tile_shape)
        num_prev_subtiles = tile_scheduler.num_tiles_executed * epi_tile_num

        def tma_store_fn(src_idx, dst_idx):
            # Fence and barrier to make sure shared memory store is visible to TMA store
            cute.arch.fence_proxy(
                cute.arch.ProxyKind.async_shared, space=cute.arch.SharedSpace.shared_cta
            )
            epilogue_barrier.arrive_and_wait()
            # Copy from shared memory to global memory
            if is_tma_warp:
                copy_D(src_idx=src_idx, dst_idx=dst_idx)
            # Can't use if statement here, epi_store_pipeline object isn't captured somehow
            if_generate(is_tma_warp, lambda: epi_store_pipeline.producer_commit())
            if_generate(is_tma_warp, lambda: epi_store_pipeline.producer_acquire())
            epilogue_barrier.arrive_and_wait()

        for epi_idx in cutlass.range_constexpr(epi_tile_num):
            # The global memory coordinate for the current epi tile
            gmem_coord = epi_tile_layout.get_hier_coord(epi_idx)
            # Copy from acc to D registers
            load_acc_subtile(tRS_rD, epi_idx)
            epi_buffer = (num_prev_subtiles + epi_idx) % self.epi_stage
            # Copy from D registers to shared memory
            copy_utils_cvt_copy(tiled_copy_r2s, tRS_rD, tRS_sD[None, None, None, epi_buffer])
            tma_store_fn(src_idx=epi_buffer, dst_idx=gmem_coord)

        return epi_read_state, epi_producer_state

    def _setup_attributes(self):
        self.tiled_mma = sm100_utils.make_trivial_tiled_mma(
            self.a_dtype,
            self.a_major_mode,
            self.b_major_mode,
            self.acc_dtype,
            self.cta_group,
            self.mma_tiler[:2],
        )

        self.mma_tiler = self.mma_tiler[:2] + (256 // self.a_dtype.width * 4,)

        self.cta_tile_shape_mnk = self.mma_tiler

        self.cluster_layout_vmnk = cute.tiled_divide(
            cute.make_layout(self.cluster_shape_mnk),
            (self.tiled_mma.thr_id.shape,),
        )

        self.epi_tile = sm100_utils.compute_epilogue_tile_shape(
            self.cta_tile_shape_mnk,
            False, # use 2 cta instrs
            self.d_layout,
            self.d_dtype,
            layout_c=None,
            elem_ty_c=None,
        )

        (
            self.num_acc_stage,
            self.ab_stage,
            self.epi_stage,
        ) = self._compute_stages(
            self.tiled_mma,
            self.mma_tiler,
            self.cta_tile_shape_mnk,
            self.epi_tile,
            self.a_dtype,
            self.b_dtype,
            self.d_dtype,
            self.d_layout,
            cutlass.utils.get_smem_capacity_in_bytes(f"sm_100"),  # smem_capacity
        )

        self.a_smem_layout_staged = sm100_utils.make_smem_layout_a(
            self.tiled_mma, self.mma_tiler, self.a_dtype, self.ab_stage
        )
        self.a_smem_load_layout_staged = self.a_smem_layout_staged
        self.b_smem_layout_staged = sm100_utils.make_smem_layout_b(
            self.tiled_mma, self.mma_tiler, self.b_dtype, self.ab_stage
        )
        self.epi_smem_layout_staged = sm100_utils.make_smem_layout_epi(
            self.d_dtype, self.d_layout, self.epi_tile, self.epi_stage
        )
        self.num_tmem_alloc_cols = self._compute_num_tmem_alloc_cols(
            self.tiled_mma, self.mma_tiler, self.num_acc_stage
        )

    @cute.jit
    def __call__(
        self,
        mA: cute.Tensor,
        mB: cute.Tensor,
        mD: cute.Tensor,
        max_active_clusters: int,
        stream: cuda.CUstream,
    ):
        self.a_dtype = mA.element_type
        self.b_dtype = mB.element_type
        self.d_dtype = mD.element_type

        self.d_layout = LayoutEnum.from_tensor(mD)
        self.a_major_mode = LayoutEnum.from_tensor(mA).mma_major_mode()
        self.b_major_mode = LayoutEnum.from_tensor(mB).mma_major_mode()

        new_stride = lambda t: tuple(
            cute.assume(s, divby=128 // t.element_type.width) if not cute.is_static(s) else s
            for s in t.stride
        )
        mA, mD = [
            cute.make_tensor(t.iterator, cute.make_layout(t.shape, stride=new_stride(t)))
            for t in (mA, mD)
        ]

        self._setup_attributes()

        a_smem_layout = cute.slice_(self.a_smem_layout_staged, (None, None, None, 0))
        b_smem_layout = cute.slice_(self.b_smem_layout_staged, (None, None, None, 0))
        a_op = sm100_utils.cluster_shape_to_tma_atom_A(
            self.cluster_shape_mnk, self.tiled_mma.thr_id
        )
        tma_atom_a, tma_tensor_a = cute.nvgpu.make_tiled_tma_atom_A(
            a_op,
            mA,
            a_smem_layout,
            self.mma_tiler,
            self.tiled_mma,
            self.cluster_layout_vmnk.shape,
            internal_type=None,
        )
        b_op = sm100_utils.cluster_shape_to_tma_atom_B(
            self.cluster_shape_mnk, self.tiled_mma.thr_id
        )
        tma_atom_b, tma_tensor_b = cute.nvgpu.make_tiled_tma_atom_B(
            b_op,
            mB,
            b_smem_layout,
            self.mma_tiler,
            self.tiled_mma,
            self.cluster_layout_vmnk.shape,
            internal_type=None,
        )

        self.num_tma_load_bytes = cute.size_in_bytes(self.b_dtype, b_smem_layout)
        self.num_tma_load_bytes += cute.size_in_bytes(self.a_dtype, a_smem_layout)

        tma_atom_d, tma_tensor_d = self._make_tma_epi_atoms_and_tensors(
            mD,
            self.epi_smem_layout_staged,
            self.epi_tile,
            op_type="store"
        )
        epilogue_params = ParamsBase()

        tile_sched_params = self.get_scheduler_params(mA, mB, mD)
        grid = TileScheduler.get_grid_shape(
            tile_sched_params, max_active_clusters
        )

        epi_smem_size = cute.cosize(self.epi_smem_layout_staged)

        @cute.struct
        class SharedStorage:
            ab_pipeline_array_ptr: cute.struct.MemRange[cutlass.Int64, self.ab_stage * 2]
            acc_pipeline_array_ptr: cute.struct.MemRange[cutlass.Int64, self.num_acc_stage * 2]
            sD: cute.struct.Align[
                cute.struct.MemRange[self.d_dtype, epi_smem_size],
                self.buffer_align_bytes,
            ]
            sA: cute.struct.Align[
                cute.struct.MemRange[self.a_dtype, cute.cosize(self.a_smem_layout_staged.outer)],
                self.buffer_align_bytes,
            ]
            sB: cute.struct.Align[
                cute.struct.MemRange[self.b_dtype, cute.cosize(self.b_smem_layout_staged.outer)],
                self.buffer_align_bytes,
            ]
            tmem_holding_buf: Int32

        self.shared_storage_class = SharedStorage

        self.kernel(
            self.tiled_mma,
            tma_atom_a,
            tma_tensor_a,
            tma_atom_b,
            tma_tensor_b,
            tma_atom_d,
            tma_tensor_d,
            epilogue_params,
            self.cluster_layout_vmnk,
            self.a_smem_layout_staged,
            self.a_smem_load_layout_staged,
            self.b_smem_layout_staged,
            self.epi_smem_layout_staged,
            self.epi_tile,
            tile_sched_params,
        ).launch(
            grid=grid,
            block=[self.threads_per_cta, 1, 1],
            cluster=self.cluster_shape_mnk,
            stream=stream,
            min_blocks_per_mp=1,
        )

    @cute.kernel
    def kernel(
        self,
        tiled_mma: cute.TiledMma,
        tma_atom_a: cute.CopyAtom | None,
        mA_mkl: cute.Tensor,
        tma_atom_b: cute.CopyAtom,
        mB_nkl: cute.Tensor,
        tma_atom_d: cute.CopyAtom,
        mD_mnl: cute.Tensor,
        epilogue_params: ParamsBase,
        cluster_layout_vmnk: cute.Layout,
        a_smem_layout: cute.ComposedLayout,
        a_smem_load_layout: cute.ComposedLayout,
        b_smem_layout: cute.ComposedLayout,
        epi_smem_layout: cute.Layout | cute.ComposedLayout | None,
        epi_tile: cute.Tile,
        tile_sched_params: ParamsBase,
    ):
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())

        if warp_idx == self.ab_load_warp_id:
            for tma_atom in (
                tma_atom_a,
                tma_atom_b,
                tma_atom_d,
            ):
                cpasync.prefetch_descriptor(tma_atom)

        cta_rank_in_cluster = cute.arch.make_warp_uniform(cute.arch.block_idx_in_cluster())
        tidx, _, _ = cute.arch.thread_idx()

        smem = cutlass.utils.SmemAllocator()
        storage = smem.allocate(self.shared_storage_class)

        tmem_holding_buf = storage.tmem_holding_buf

        ab_pipeline = self.make_ab_pipeline(
            tiled_mma=tiled_mma,
            cluster_layout_vmnk=cluster_layout_vmnk,
            ab_pipeline_mbar_ptr=storage.ab_pipeline_array_ptr.data_ptr(),
        )
        acc_pipeline = self.make_acc_pipeline(
            cluster_layout_vmnk=cluster_layout_vmnk,
            acc_pipeline_mbar_ptr=storage.acc_pipeline_array_ptr.data_ptr(),
        )

        sA_mma = storage.sA.get_tensor(a_smem_layout.outer, swizzle=a_smem_layout.inner)
        sA = storage.sA.get_tensor(a_smem_load_layout.outer, swizzle=a_smem_load_layout.inner)
        sB = storage.sB.get_tensor(b_smem_layout.outer, swizzle=b_smem_layout.inner)
        sD = storage.sD.get_tensor(epi_smem_layout.outer, swizzle=epi_smem_layout.inner)

        thr_mma = tiled_mma.get_slice(0)

        acc_shape = tiled_mma.partition_shape_C(self.mma_tiler[:2])
        tCtAcc_fake = tiled_mma.make_fragment_C(cute.append(acc_shape, self.num_acc_stage))

        len_m_static = Int32(mA_mkl.shape[0])
        len_k_static = Int32(mA_mkl.shape[1])

        tile_scheduler = TileScheduler.create(tile_sched_params)

        tmem_alloc_barrier = cutlass.pipeline.NamedBarrier(
            barrier_id=int(NamedBarrierEnum.TmemPtr),
            num_threads=cute.arch.WARP_SIZE * (self.num_epi_warps + 1),
        )

        if warp_idx == self.ab_load_warp_id:
            is_tma_warp = True
            block_in_cluster_coord_vmnk = cluster_layout_vmnk.get_flat_coord(cta_rank_in_cluster)

            work_tile = tile_scheduler.get_current_work()
            ab_producer_state = cutlass.pipeline.make_pipeline_state(
                cutlass.pipeline.PipelineUserType.Producer, self.ab_stage
            )
            while work_tile.is_valid_tile:
                tile_coord_mnkl = work_tile.tile_idx
                batch_idx = tile_coord_mnkl[3]
                mma_tile_coord_mnl = (
                    tile_coord_mnkl[0],
                    tile_coord_mnkl[1],
                    tile_coord_mnkl[3],
                )
                mA_mk = mA_mkl[None, None, batch_idx]
                gA_mk = cute.local_tile(
                    mA_mk,
                    cute.select(self.mma_tiler, [0, 2]),
                    (mma_tile_coord_mnl[0], None),
                )

                gB_nk = cute.local_tile(
                    mB_nkl[None, None, batch_idx],
                    cute.select(self.mma_tiler, [1, 2]),
                    (mma_tile_coord_mnl[1], None),
                )

                len_k = len_k_static
                a_cta_layout = cute.make_layout(
                    cute.slice_(cluster_layout_vmnk, (0, 0, None, 0)).shape
                )
                tCgA = thr_mma.partition_A(gA_mk)
                copy_A = copy_utils_tma_get_copy_fn(
                    tma_atom_a,
                    cta_coord=block_in_cluster_coord_vmnk[2],
                    cta_layout=a_cta_layout,
                    src_tensor=tCgA,
                    dst_tensor=sA,
                    mcast_mask=None,
                )

                tCgB = thr_mma.partition_B(gB_nk)
                copy_B = copy_utils_tma_get_copy_fn(
                    tma_atom_b,
                    cta_coord=block_in_cluster_coord_vmnk[1],
                    cta_layout=cute.make_layout(
                        cute.slice_(cluster_layout_vmnk, (0, None, 0, 0)).shape
                    ),
                    src_tensor=tCgB,
                    dst_tensor=sB,
                    mcast_mask=None,
                )
                k_tile_cnt = cute.ceil_div(len_k, self.cta_tile_shape_mnk[2])
                ab_producer_state = self.load_AB(
                    ab_pipeline,
                    ab_producer_state,
                    copy_A,
                    copy_B,
                    k_tile_cnt,
                )
                tile_scheduler.advance_to_next_work()
                work_tile = tile_scheduler.get_current_work()
            ab_pipeline.producer_tail(ab_producer_state)

        if warp_idx == self.mma_warp_id:
            tmem_alloc_barrier.arrive_and_wait()
            acc_tmem_ptr = cute.arch.retrieve_tmem_ptr(
                self.acc_dtype, alignment=16, ptr_to_buffer_holding_addr=tmem_holding_buf
            )
            tCrA = tiled_mma.make_fragment_A(sA_mma)
            tCrB = tiled_mma.make_fragment_B(sB)
            tCtAcc_base = cute.make_tensor(acc_tmem_ptr, tCtAcc_fake.layout)

            work_tile = tile_scheduler.get_current_work()
            ab_consumer_state = cutlass.pipeline.make_pipeline_state(
                cutlass.pipeline.PipelineUserType.Consumer, self.ab_stage
            )
            acc_producer_state = cutlass.pipeline.make_pipeline_state(
                cutlass.pipeline.PipelineUserType.Producer, self.num_acc_stage
            )
            while work_tile.is_valid_tile:
                # Get tile coord from tile scheduler
                tile_coord_mnkl = work_tile.tile_idx
                batch_idx = tile_coord_mnkl[3]
                k_len = len_k_static
                k_tile_cnt = cute.ceil_div(k_len, self.mma_tiler[2])
                tCtAcc = tCtAcc_base[None, None, None, acc_producer_state.index]
                ab_consumer_state, acc_producer_state, tiled_mma = self.mma(
                    ab_pipeline,
                    acc_pipeline,
                    ab_consumer_state,
                    acc_producer_state,
                    tiled_mma,
                    tCrA,
                    tCrB,
                    tCtAcc,
                    k_tile_cnt,
                    cta_rank_in_cluster,
                )
                # Advance to next tile
                tile_scheduler.advance_to_next_work()
                work_tile = tile_scheduler.get_current_work()

            # Wait for accumulator buffer empty
            acc_pipeline.producer_tail(acc_producer_state)

        # Specialized epilogue warps
        if warp_idx < self.mma_warp_id:
            # Alloc tensor memory buffer
            if warp_idx == self.epilog_warp_id[0]:
                cute.arch.alloc_tmem(
                    self.num_tmem_alloc_cols, tmem_holding_buf, is_two_cta=False
                )
            # Bar sync for retrieve tensor memory ptr from shared memory
            tmem_alloc_barrier.arrive_and_wait()

            is_tma_warp = Boolean(warp_idx == self.epilog_warp_id[0])

            # Retrieving tensor memory ptr and make accumulator tensor
            acc_tmem_ptr = cute.arch.retrieve_tmem_ptr(
                self.acc_dtype, alignment=16, ptr_to_buffer_holding_addr=tmem_holding_buf
            )
            # (MMA, MMA_M, MMA_N, STAGE)
            tCtAcc_base = cute.make_tensor(acc_tmem_ptr, tCtAcc_fake.layout)

            epilogue_barrier = cutlass.pipeline.NamedBarrier(
                barrier_id=int(NamedBarrierEnum.Epilogue),
                num_threads=self.num_epi_warps * cute.arch.WARP_SIZE,
            )

            # Partition for epilogue
            epi_tidx = tidx
            tiled_copy_t2r, tTR_tAcc_base, tTR_rAcc = self.epilog_tmem_copy_and_partition(
                epi_tidx, tCtAcc_base, epi_tile,
            )

            tTR_rD = cute.make_fragment(tTR_rAcc.shape, self.acc_dtype)
            tiled_copy_r2s, tRS_rD, tRS_sD = self.epilog_smem_store_and_partition(
                tiled_copy_t2r, self.d_layout, self.d_dtype, tTR_rD, sD, epi_tidx
            )

            # Persistent tile scheduling loop
            work_tile = tile_scheduler.get_current_work()
            acc_consumer_state = cutlass.pipeline.make_pipeline_state(
                cutlass.pipeline.PipelineUserType.Consumer, self.num_acc_stage
            )
            epi_store_pipeline = self.make_epi_store_pipeline()
            epi_read_state = cutlass.pipeline.make_pipeline_state(
                cutlass.pipeline.PipelineUserType.Consumer, 0
            )
            while work_tile.is_valid_tile:
                # Get tile coord from tile scheduler
                tile_coord_mnkl = work_tile.tile_idx
                batch_idx = tile_coord_mnkl[3]

                tTR_tAcc = tTR_tAcc_base[None, None, None, None, None, acc_consumer_state.index]

                acc_pipeline.consumer_wait(acc_consumer_state)

                copy_D = self.epilog_gmem_copy_and_partition(
                    tma_atom_d,
                    mD_mnl[None, None, batch_idx],
                    self.cta_tile_shape_mnk[:2],
                    epi_tile,
                    sD,
                    tile_coord_mnkl,
                )

                tTR_tAcc = cute.group_modes(tTR_tAcc, 3, cute.rank(tTR_tAcc))
                k_len = len_k_static
                load_acc_subtile = partial(
                    self.epi_load_acc_subtile,
                    tiled_copy_t2r,
                    tiled_copy_r2s,
                    tTR_tAcc,
                    tTR_rAcc,
                )

                epi_read_state, _ = self.epilogue(
                    epilogue_params,
                    epi_store_pipeline,
                    epi_read_state,
                    None,  # epi_producer_state
                    epi_tile,
                    load_acc_subtile,
                    tRS_rD,
                    tiled_copy_t2r,
                    tiled_copy_r2s,
                    tRS_sD,
                    copy_D,
                    tile_coord_mnkl,
                    epilogue_barrier,
                    tile_scheduler,
                    epi_tidx,
                    is_tma_warp,
                )

                # Async arrive accumulator buffer empty
                with cute.arch.elect_one():
                    acc_pipeline.consumer_release(acc_consumer_state)
                acc_consumer_state.advance()

                # Advance to next tile
                tile_scheduler.advance_to_next_work()
                work_tile = tile_scheduler.get_current_work()

            # Dealloc the tensor memory buffer
            if warp_idx == self.epilog_warp_id[0]:
                cute.arch.relinquish_tmem_alloc_permit(is_two_cta=False)
            epilogue_barrier.arrive_and_wait()
            if warp_idx == self.epilog_warp_id[0]:
                cute.arch.dealloc_tmem(
                    acc_tmem_ptr, self.num_tmem_alloc_cols, is_two_cta=False
                )

            # Wait for D store complete
            if is_tma_warp:
                epi_store_pipeline.producer_tail()

    @cute.jit
    def mma(
        self,
        ab_pipeline: cutlass.pipeline.PipelineAsync,
        acc_pipeline: cutlass.pipeline.PipelineAsync,
        ab_consumer_state: cutlass.pipeline.PipelineState,
        acc_producer_state: cutlass.pipeline.PipelineState,
        tiled_mma: cute.TiledMma,
        tCrA: cute.Tensor,
        tCrB: cute.Tensor,
        acc: cute.Tensor,
        k_tile_cnt: Int32,
        cta_rank_in_cluster: Int32,
    ) -> tuple[cutlass.pipeline.PipelineState, cutlass.pipeline.PipelineState, cute.TiledMma]:
        # Peek (try_wait) AB buffer full for k_tile = 0
        peek_ab_full_status = Boolean(True)
        if k_tile_cnt > 0:
            peek_ab_full_status = ab_pipeline.consumer_try_wait(ab_consumer_state)
        # Wait for accumulator buffer empty
        acc_pipeline.producer_acquire(acc_producer_state)
        # Reset the ACCUMULATE field for each tile
        tiled_mma.set(tcgen05.Field.ACCUMULATE, False)
        # Mma mainloop
        num_k_blocks = cute.size(tCrA, mode=[2])
        for k_tile in cutlass.range(k_tile_cnt, unroll=1):
            # Conditionally wait for AB buffer full
            ab_pipeline.consumer_wait(ab_consumer_state, peek_ab_full_status)
            for k_blk_idx in cutlass.range(num_k_blocks, unroll_full=True):
                k_blk_coord = (None, None, k_blk_idx, ab_consumer_state.index)
                cute.gemm(tiled_mma, acc, tCrA[k_blk_coord], tCrB[k_blk_coord], acc)
                tiled_mma.set(tcgen05.Field.ACCUMULATE, True)
            # Async arrive AB buffer empty
            ab_pipeline.consumer_release(ab_consumer_state)
            ab_consumer_state.advance()
            # Peek (try_wait) AB buffer full for k_tile = k_tile + 1
            peek_ab_full_status = Boolean(True)
            if k_tile + 1 < k_tile_cnt:
                peek_ab_full_status = ab_pipeline.consumer_try_wait(ab_consumer_state)
        acc_pipeline.producer_commit(acc_producer_state)
        acc_producer_state.advance()
        return ab_consumer_state, acc_producer_state, tiled_mma

    @cute.jit
    def epi_load_acc_subtile(
        self,
        tiled_copy_t2r: cute.TiledCopy,
        tiled_copy_r2s: cute.TiledCopy,
        tTR_tAcc: cute.Tensor,
        tTR_rAcc: cute.Tensor,
        tRS_rD: cute.Tensor,
        epi_idx: int,
    ):
        cute.copy(tiled_copy_t2r, tTR_tAcc[None, None, None, epi_idx], tTR_rAcc)
        tRS_rAcc = tiled_copy_r2s.retile(tTR_rAcc)
        tRS_rD.store(tRS_rAcc.load())

    def epilog_tmem_copy_and_partition(
        self,
        tidx: Int32,
        tAcc: cute.Tensor,
        epi_tile: cute.Tile,
    ) -> tuple[cute.TiledCopy, cute.Tensor, cute.Tensor]:
        copy_atom_t2r = sm100_utils.get_tmem_load_op(
            self.cta_tile_shape_mnk,
            self.d_layout,
            self.d_dtype,
            self.acc_dtype,
            epi_tile,
            False, # use 2cta instrs,
        )
        tAcc_epi = cute.flat_divide(tAcc[((None, None), 0, 0, None)], epi_tile)
        tiled_copy_t2r = tcgen05.make_tmem_copy(copy_atom_t2r, tAcc_epi[(None, None, 0, 0, 0)])

        thr_copy_t2r = tiled_copy_t2r.get_slice(tidx)
        tTR_tAcc = thr_copy_t2r.partition_S(tAcc_epi)

        cAcc = cute.make_identity_tensor((self.cta_tile_shape_mnk[0], self.cta_tile_shape_mnk[1]))
        cAcc_epi = cute.flat_divide(cAcc, epi_tile)
        tTR_cAcc = thr_copy_t2r.partition_D(cAcc_epi)
        tTR_rAcc = cute.make_fragment(tTR_cAcc[None, None, None, 0, 0].shape, self.acc_dtype)
        return tiled_copy_t2r, tTR_tAcc, tTR_rAcc

    def epilog_smem_store_and_partition(
        self,
        tiled_copy_t2r: cute.TiledCopy,
        d_layout: LayoutEnum,
        dtype: Type[cutlass.Numeric],
        tTR_rD: cute.Tensor,
        sD: cute.Tensor,
        tidx: Int32,
    ) -> tuple[cute.TiledCopy, cute.Tensor, cute.Tensor]:
        copy_atom_r2s = sm100_utils.get_smem_store_op(
            d_layout,
            dtype,
            self.acc_dtype,
            tiled_copy_t2r,
        )
        tiled_copy_r2s = cute.make_tiled_copy_D(copy_atom_r2s, tiled_copy_t2r)
        thr_copy_r2s = tiled_copy_r2s.get_slice(tidx)
        tRS_sD = thr_copy_r2s.partition_D(sD)
        tRS_rD = tiled_copy_r2s.retile(tTR_rD)
        return tiled_copy_r2s, tRS_rD, tRS_sD

    def make_epi_store_pipeline(self):
        num_epi_threads = self.num_epi_warps * cute.arch.WARP_SIZE
        epi_store_producer_group = cutlass.pipeline.CooperativeGroup(cutlass.pipeline.Agent.Thread, num_epi_threads)
        return cutlass.pipeline.PipelineTmaStore.create(
            num_stages=self.epi_stage, producer_group=epi_store_producer_group
        )

    @cute.jit
    def make_ab_pipeline(
        self,
        tiled_mma: cute.TiledMma,
        cluster_layout_vmnk: cute.Layout,
        ab_pipeline_mbar_ptr: cute.Pointer,
    ) -> cutlass.pipeline.PipelineAsync:
        ab_pipeline_producer_group = cutlass.pipeline.CooperativeGroup(cutlass.pipeline.Agent.Thread)
        ab_pipeline_consumer_group = cutlass.pipeline.CooperativeGroup(cutlass.pipeline.Agent.Thread)
        return cutlass.pipeline.PipelineTmaUmma.create(
            barrier_storage=ab_pipeline_mbar_ptr,
            num_stages=self.ab_stage,
            producer_group=ab_pipeline_producer_group,
            consumer_group=ab_pipeline_consumer_group,
            tx_count=self.num_tma_load_bytes,
            cta_layout_vmnk=cluster_layout_vmnk,
        )

    def make_acc_pipeline(
        self, cluster_layout_vmnk: cute.Layout, acc_pipeline_mbar_ptr: cute.Pointer
    ) -> cutlass.pipeline.PipelineAsync:
        acc_pipeline_producer_group = cutlass.pipeline.CooperativeGroup(cutlass.pipeline.Agent.Thread)
        acc_pipeline_consumer_group = cutlass.pipeline.CooperativeGroup(
            cutlass.pipeline.Agent.Thread, self.num_epi_warps,
        )
        return cutlass.pipeline.PipelineUmmaAsync.create(
            barrier_storage=acc_pipeline_mbar_ptr,
            num_stages=self.num_acc_stage,
            producer_group=acc_pipeline_producer_group,
            consumer_group=acc_pipeline_consumer_group,
            cta_layout_vmnk=cluster_layout_vmnk,
        )

    @classmethod
    def _compute_stages(
        cls,
        tiled_mma: cute.TiledMma,
        mma_tiler_mnk: tuple[int, int, int],
        cta_tile_shape_mnk: tuple[int, int, int],
        epi_tile: cute.Tile,
        a_dtype: Type[cutlass.Numeric],
        b_dtype: Type[cutlass.Numeric],
        d_dtype: Type[cutlass.Numeric],
        d_layout: LayoutEnum,
        smem_capacity: int,
    ) -> tuple[int, int, int]:
        # Default ACC stages
        num_acc_stage = 2

        # Default D stages
        epi_stage = 4 if cute.size(epi_tile[1]) <= 16 else 2

        # Calculate smem layout and size for one stage of A, B, and C
        a_smem_layout_staged_one = sm100_utils.make_smem_layout_a(
            tiled_mma,
            mma_tiler_mnk,
            a_dtype,
            1,  # a tmp 1 stage is provided
        )
        b_smem_layout_staged_one = sm100_utils.make_smem_layout_b(
            tiled_mma,
            mma_tiler_mnk,
            b_dtype,
            1,  # a tmp 1 stage is provided
        )
        d_smem_layout_staged_one = sm100_utils.make_smem_layout_epi(d_dtype, d_layout, epi_tile, 1)

        ab_bytes_per_stage = cute.size_in_bytes(
            a_dtype, a_smem_layout_staged_one
        ) + cute.size_in_bytes(b_dtype, b_smem_layout_staged_one)
        mbar_helpers_bytes = 1024
        d_bytes_per_stage = cute.size_in_bytes(d_dtype, d_smem_layout_staged_one)
        epi_bytes_per_stage = d_bytes_per_stage
        epi_bytes = epi_bytes_per_stage * epi_stage

        remaining_bytes = smem_capacity - mbar_helpers_bytes - epi_bytes
        ab_stage = remaining_bytes // ab_bytes_per_stage

        epi_stage += (remaining_bytes - ab_bytes_per_stage * ab_stage) // (epi_bytes_per_stage)
        return num_acc_stage, ab_stage, epi_stage

    @staticmethod
    def _compute_num_tmem_alloc_cols(
        tiled_mma: cute.TiledMma,
        mma_tiler: tuple[int, int, int],
        num_acc_stage: int,
    ) -> int:
        acc_shape = tiled_mma.partition_shape_C(mma_tiler[:2])
        tCtAcc_fake = tiled_mma.make_fragment_C(cute.append(acc_shape, num_acc_stage))
        num_tmem_alloc_cols = cutlass.utils.get_num_tmem_alloc_cols(tCtAcc_fake)
        return num_tmem_alloc_cols

if __name__ == "__main__":
    main()
