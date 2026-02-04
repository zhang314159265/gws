from enum import IntEnum
import enum

# ============ XXX SPLIT ===========

class RasterOrder(IntEnum):
    AlongM = 0
    AlongN = 1

class RasterOrderOption(IntEnum):
    AlongM = 0
    AlongN = 1
    Heuristic = 2  # Pick AlongM if tiles_n > tiles_m, else AlongN

class NamedBarrierGemm(enum.IntEnum):
    Epilogue = enum.auto()  # starts from 1 as barrier 0 is reserved for sync_threads()
    # For mainloop load warps to signal that the epilogue load warp can start.
    # This is to avoid loading C too early, interfering with loading A and B.
    EpilogueLoad = enum.auto()
    MmaWG0 = enum.auto()
    MmaWG1 = enum.auto()
    EpiWG0 = enum.auto()
    EpiWG1 = enum.auto()
    TmemPtr = enum.auto()





# Based on the cute-dsl example:
# https://github.com/NVIDIA/cutlass/blob/main/examples/python/CuTeDSL/blackwell/dense_gemm_persistent.py

import argparse
from typing import Optional, Type, Tuple, Union, Callable, Literal
from functools import partial
from cutlass.cutlass_dsl import if_generate

import cuda.bindings.driver as cuda
import torch

import cutlass
import cutlass.cute as cute
from cutlass.cute.nvgpu import cpasync, tcgen05
import cutlass.torch as cutlass_torch
import cutlass.pipeline as pipeline
import cutlass.utils.blackwell_helpers as sm100_utils
import cutlass.utils.blockscaled_layout as blockscaled_utils
from cutlass.cute.nvgpu.warp import (
    LdMatrix8x8x16bOp,
    LdMatrix16x16x8bOp,
    StMatrix8x8x16bOp,
    StMatrix16x8x8bOp,
)
from cutlass import Int32, Float32, Boolean, const_expr, Int32, Uint32
from cutlass.utils import LayoutEnum
from cutlass.cute.runtime import from_dlpack, make_ptr
from cutlass.base_dsl.typing import JitArgument
from dataclasses import dataclass, fields
from cutlass.cutlass_dsl import NumericMeta
from cutlass.cutlass_dsl import T, dsl_user_op
from cutlass._mlir.dialects import llvm

@dsl_user_op
def umulhi(a: Int32, b: Int32, *, loc=None, ip=None) -> Uint32:
    return Uint32(
        llvm.inline_asm(
            T.i32(),
            [Int32(a).ir_value(loc=loc, ip=ip), Int32(b).ir_value(loc=loc, ip=ip)],
            "mul.hi.u32 $0, $1, $2;",
            "=r,r,r",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
        )
    )



def find_log2(x: Int32) -> Int32:
    a: Int32 = Int32(31 - clz(x))
    return a + ((x & (x - 1)) != 0)  # Round up, add 1 if not a power of 2.
@cute.jit
def clz(x: Int32) -> Int32:
    # for i in cutlass.range_constexpr(32):
    #     if (1 << (31 - i)) & x:
    #         return Int32(i)
    # return Int32(32)
    # Early exit is not supported yet
    res = Int32(32)
    done = False
    for i in cutlass.range(32):
        if ((1 << (31 - i)) & x) and not done:
            res = Int32(i)
            done = True
    return res




StaticTypes = (cutlass.Constexpr, NumericMeta, int, bool, str, float, type(None))

@dataclass
class ParamsBase:
    def __extract_mlir_values__(self):
        all_fields = [getattr(self, field.name) for field in fields(self)]
        non_constexpr_fields = [f for f in all_fields if not isinstance(f, StaticTypes)]
        values, self._values_pos = [], []
        for obj in non_constexpr_fields:
            obj_values = cutlass.extract_mlir_values(obj)
            values += obj_values
            self._values_pos.append(len(obj_values))
        return values

    def __new_from_mlir_values__(self, values):
        all_fields = {field.name: getattr(self, field.name) for field in fields(self)}
        constexpr_fields = {n: f for n, f in all_fields.items() if isinstance(f, StaticTypes)}
        non_constexpr_fields = {
            n: f for n, f in all_fields.items() if not isinstance(f, StaticTypes)
        }
        for (name, field), n_items in zip(non_constexpr_fields.items(), self._values_pos):
            non_constexpr_fields[name] = cutlass.new_from_mlir_values(field, values[:n_items])
            values = values[n_items:]
        return self.__class__(**non_constexpr_fields, **constexpr_fields)

@dataclass
class ArgumentsBase(JitArgument):
    def __c_pointers__(self):
        all_fields = [getattr(self, field.name) for field in fields(self)]
        non_constexpr_fields = [f for f in all_fields if not isinstance(f, StaticTypes)]
        c_ptrs = []
        for obj in non_constexpr_fields:
            if hasattr(obj, "__c_pointers__"):
                c_ptrs.extend(obj.__c_pointers__())
        return c_ptrs

    def __get_mlir_types__(self):
        all_fields = [getattr(self, field.name) for field in fields(self)]
        non_constexpr_fields = [f for f in all_fields if not isinstance(f, StaticTypes)]
        types, self._values_pos = [], []
        for obj in non_constexpr_fields:
            if hasattr(obj, "__get_mlir_types__"):
                obj_types = obj.__get_mlir_types__()
                types.extend(obj_types)
                self._values_pos.append(len(obj_types))
            else:
                self._values_pos.append(0)
        return types

    def __new_from_mlir_values__(self, values):
        all_fields = {field.name: getattr(self, field.name) for field in fields(self)}
        constexpr_fields = {n: f for n, f in all_fields.items() if isinstance(f, StaticTypes)}
        non_constexpr_fields = {
            n: f for n, f in all_fields.items() if not isinstance(f, StaticTypes)
        }
        for (name, field), n_items in zip(non_constexpr_fields.items(), self._values_pos):
            non_constexpr_fields[name] = cutlass.new_from_mlir_values(field, values[:n_items])
            values = values[n_items:]
        return self.__class__(**non_constexpr_fields, **constexpr_fields)

# Grouping arguments together that should be passed to __call__
@dataclass
class VarlenArguments(ArgumentsBase):
    mCuSeqlensM: Optional[cute.Tensor] = None
    mCuSeqlensK: Optional[cute.Tensor] = None
    mTensormaps: Optional[cute.Tensor] = None
    mAIdx: Optional[cute.Tensor] = None





from cutlass.pipeline import PipelineAsync, PipelineTmaAsync, PipelineState, PipelineUserType


@dsl_user_op
def copy_utils_cvt_copy(
    tiled_copy: cute.TiledCopy,
    src: cute.Tensor,
    dst: cute.Tensor,
    *,
    pred: Optional[cute.Tensor] = None,
    retile: bool = False,
    loc=None,
    ip=None,
    **kwargs,
) -> None:
    assert isinstance(src.iterator, cute.Pointer) and src.memspace == cute.AddressSpace.rmem
    if const_expr(src.element_type != dst.element_type):
        src_cvt = cute.make_fragment_like(src, dst.element_type)
        src_cvt.store(src.load().to(dst.element_type))
        src = src_cvt
    if const_expr(retile):
        src = tiled_copy.retile(src)
    cute.copy(tiled_copy, src, dst, pred=pred, loc=loc, ip=ip, **kwargs)




@cute.jit
def copy_utils_gather_k_get_copy_fn(
    thr_copy_A: cute.ThrCopy,
    mA: cute.Tensor,  # (tile_M, whatever)
    sA: cute.Tensor,  # (tile_M, tile_N, STAGE)
    gsAIdx: cute.Tensor,  # (tile_K, RestK), either gmem or smem
    limit_m: Int32,
    limit_k: Int32,
) -> Callable:
    gAIdx, sAIdx = None, None
    if const_expr(gsAIdx.memspace == cute.AddressSpace.gmem):
        gAIdx = gsAIdx
    else:
        assert gsAIdx.memspace == cute.AddressSpace.smem
        sAIdx = gsAIdx
    tile_shape_mk = (cute.size(sA, mode=[0]), cute.size(sA, mode=[1]))
    # (atom_v, CPY_M, 1, STAGE)
    tAsA = thr_copy_A.partition_D(sA)
    # m-major
    tAsA = cute.group_modes(tAsA, 0, 3)

    is_even_m_smem = tile_shape_mk[0] % thr_copy_A.tiler_mn[0].shape == 0
    if const_expr(not is_even_m_smem):
        limit_m = min(limit_m, tile_shape_mk[0])
    elems_per_load = cute.size(tAsA.shape[0][0])
    cA = cute.make_identity_tensor(tile_shape_mk)
    tAcA = thr_copy_A.partition_S(cA)
    t0AcA = thr_copy_A.get_slice(0).partition_S(cA)
    # Instead of comparing tAcA to limit_m, we instead compare t0AcA to limit_m - tAcA[0][0]
    # since we know that tAcA[m][0] = t0AcA[m][0] + tAcA[0][0].
    # This is so that when we do the comparison, t0AcA is known at compile time.
    limit_m = limit_m - tAcA[0][0]
    limit_k = limit_k - tAcA[0][1]
    # Read and cache indices for A
    rows_per_thread = const_expr(cute.size(tAcA.shape, mode=[1]))
    cols_per_thread = const_expr(cute.size(tAcA.shape, mode=[2]))
    tApA_m = cute.make_fragment(rows_per_thread, Boolean)
    for m in cutlass.range(rows_per_thread, unroll_full=True):
        tApA_m[m] = t0AcA[0, m, 0][0] < limit_m
    threads_per_col = const_expr(thr_copy_A.tiler_mn[0].shape // elems_per_load)
    # This is very convoluted but idk a better way
    # for tile_M=128, flat_divide gives (8, 16, K),
    # then logical_divide gives ((8, 1), (8, 2), K).
    tidx = thr_copy_A.thr_idx
    tAmA = cute.logical_divide(
        cute.flat_divide(mA, (elems_per_load,)), (elems_per_load, threads_per_col)
    )[None, (tidx % threads_per_col, None), None]  # ((8, 1), 2, K)

    def prefetch_from_gmem_fn(src_idx, pred: bool = False) -> Tuple[cute.Tensor, cute.Tensor]:
        # Prefetch mAIdx early, even before smem is free
        tApA_k = None
        if const_expr(pred):
            tApA_k = cute.make_fragment(cols_per_thread, Boolean)
            limit_k_cur = limit_k - src_idx * tile_shape_mk[1]
            for k in cutlass.range(cols_per_thread, unroll_full=True):
                tApA_k[k] = t0AcA[0, 0, k][1] < limit_k_cur
        gAIdx_cur = gAIdx[None, src_idx]
        k_idx = cute.make_fragment(cols_per_thread, Int32)
        for k in cutlass.range(cols_per_thread):
            col_idx = tAcA[0, 0, k][1]
            if const_expr(not pred):
                k_idx[k] = gAIdx_cur[col_idx]
            else:
                if tApA_k[k]:
                    k_idx[k] = gAIdx_cur[col_idx]
                else:
                    k_idx[k] = -1
        return k_idx, tApA_k

    def prefetch_from_smem_fn(
        a_prefetch_pipeline, src_idx, dst_idx, a_prefetch_consumer_state, pred: bool = False
    ) -> Tuple[cute.Tensor, cute.Tensor]:
        tApA_k = None
        if const_expr(pred):
            tApA_k = cute.make_fragment(cols_per_thread, Boolean)
            limit_k_cur = limit_k - src_idx * tile_shape_mk[1]
            for k in cutlass.range(cols_per_thread, unroll_full=True):
                tApA_k[k] = t0AcA[0, 0, k][1] < limit_k_cur
        a_prefetch_pipeline.consumer_wait(a_prefetch_consumer_state)
        sAIdx_cur = sAIdx[None, dst_idx]
        k_idx = cute.make_fragment(cols_per_thread, Int32)
        for k in cutlass.range(cols_per_thread):
            col_idx = tAcA[0, 0, k][1]
            k_idx[k] = sAIdx_cur[col_idx]
        cute.arch.sync_warp()
        with cute.arch.elect_one():
            a_prefetch_pipeline.consumer_release(a_prefetch_consumer_state)
        return k_idx, tApA_k

    def copy_fn(
        src_idx, dst_idx, k_idx_tApA_k: Tuple[cute.Tensor, cute.Tensor], pred: bool = False
    ):
        k_idx, tApA_k = k_idx_tApA_k
        tApA_k_pred = None
        if const_expr(pred):
            tApA_k_pred = cute.prepend_ones(tApA_k, up_to_rank=2)  # (1, cols_per_thread)
        for k in cutlass.range_constexpr(tAcA.shape[2]):
            # copy_A(tAmA[None, None, k_idx[k]], tAsA[(None, None, k), smem_idx], pred=cute.prepend_ones(tApA_m, up_to_rank=2))
            for m in cutlass.range_constexpr(tAcA.shape[1]):
                if tApA_m[m]:
                    cute.copy(
                        thr_copy_A,
                        tAmA[None, m, k_idx[k]],
                        tAsA[(None, m, k), dst_idx],
                        pred=None if const_expr(tApA_k_pred is None) else tApA_k_pred[None, k],
                    )

    return copy_fn, prefetch_from_gmem_fn if const_expr(
        gAIdx is not None
    ) else prefetch_from_smem_fn

@cute.jit
def copy_utils_gather_m_get_copy_fn(
    thr_copy_A: cute.ThrCopy,
    mA: cute.Tensor,  # (whatever, K)
    sA: cute.Tensor,  # (tile_M, tile_N, STAGE)
    gsAIdx: cute.Tensor,  # (tile_M), either gmem or smem
    limit_m: Int32,
    limit_k: Int32,
) -> Callable:
    tile_shape_mk = (cute.size(sA, mode=[0]), cute.size(sA, mode=[1]))
    tAsA = thr_copy_A.partition_D(sA)
    # k-major
    assert tAsA.shape[2] == 1
    tAsA = cute.group_modes(cute.slice_(tAsA, (None, None, 0, None)), 0, 2)

    is_even_m_smem = tile_shape_mk[0] % thr_copy_A.tiler_mn[0].shape == 0
    if const_expr(not is_even_m_smem):
        limit_m = min(limit_m, tile_shape_mk[0])
    elems_per_load = cute.size(tAsA.shape[0][0])
    cA = cute.make_identity_tensor(tile_shape_mk)
    tAcA = thr_copy_A.partition_S(cA)
    t0AcA = thr_copy_A.get_slice(0).partition_S(cA)
    # Instead of comparing tAcA to limit_m, we instead compare t0AcA to limit_m - tAcA[0][0]
    # since we know that tAcA[m][0] = t0AcA[m][0] + tAcA[0][0].
    # This is so that when we do the comparison, t0AcA is known at compile time.
    limit_m = limit_m - tAcA[0][0]
    limit_k = limit_k - tAcA[0][1]
    # Read and cache indices for A
    rows_per_thread = const_expr(cute.size(tAcA.shape, mode=[1]))
    cols_per_thread = const_expr(cute.size(tAcA.shape, mode=[2]))
    tApA_m = cute.make_fragment(rows_per_thread, Boolean)
    for m in cutlass.range(rows_per_thread, unroll_full=True):
        tApA_m[m] = t0AcA[0, m, 0][0] < limit_m
    m_idx = cute.make_fragment(rows_per_thread, Int32)
    for m in cutlass.range(rows_per_thread, unroll_full=True):
        row_idx = tAcA[0, m, 0][0]
        if tApA_m[m]:
            m_idx[m] = gsAIdx[row_idx]
        else:
            m_idx[m] = 0  # It's ok to load row 0 in the case of OOB

    mA_k = cute.logical_divide(mA, (None, tile_shape_mk[1]))

    def copy_fn(src_idx, dst_idx, pred: bool = False):
        tApA_k = None
        if const_expr(pred):
            tApA_k = cute.make_fragment(cols_per_thread, Boolean)
            limit_k_cur = limit_k - src_idx * tile_shape_mk[1]
            for k in cutlass.range(cols_per_thread, unroll_full=True):
                tApA_k[k] = t0AcA[0, 0, k][1] < limit_k_cur
        mA_cur = mA_k[None, (None, src_idx)]
        for m in cutlass.range_constexpr(tAcA.shape[1]):
            # cute.tiled_divide(mA_cur[m_idx[m], None], (elems_per_load,)) would give shape
            # ((elems_per_load), thread_per_row)
            # But we actually want shape ((elems_per_load, 1), thread_per_row) to match tAsA
            # So we append 1s to the last dimension and then do tiled_divide, then slice.
            mA_row = cute.tiled_divide(
                cute.append_ones(mA_cur[m_idx[m], None], up_to_rank=2), (elems_per_load, 1)
            )[None, None, 0]
            if const_expr(is_even_m_smem) or tApA_m[m]:
                # There's only 1 load per row
                assert cute.size(tAcA.shape, mode=[2]) == 1
                ki = tAcA[0, 0, 0][1] // elems_per_load
                cute.copy(thr_copy_A, mA_row[None, ki], tAsA[(None, m), dst_idx], pred=tApA_k)

    return copy_fn



def copy_utils_tma_get_copy_fn(
    atom: cute.CopyAtom,
    cta_coord: cute.Coord,
    cta_layout: cute.Layout,
    src_tensor: cute.Tensor,
    dst_tensor: cute.Tensor,
    filter_zeros: bool = False,
    single_stage: bool = False,
    **kwargs,
) -> Callable:
    src_is_smem = const_expr(
        isinstance(src_tensor.iterator, cute.Pointer)
        and src_tensor.memspace == cute.AddressSpace.smem
    )
    smem_tensor, gmem_tensor = (src_tensor, dst_tensor) if src_is_smem else (dst_tensor, src_tensor)
    group_rank_smem = const_expr(cute.rank(smem_tensor) - (1 if not single_stage else 0))
    group_rank_gmem = const_expr(cute.rank(gmem_tensor) - (1 if not single_stage else 0))
    # ((atom_v, rest_v), STAGE), ((atom_v, rest_v), RestK)
    s, g = cpasync.tma_partition(
        atom,
        cta_coord,
        cta_layout,
        cute.group_modes(smem_tensor, 0, group_rank_smem),
        cute.group_modes(gmem_tensor, 0, group_rank_gmem),
    )
    if const_expr(filter_zeros):
        s = cute.filter_zeros(s)
        g = cute.filter_zeros(g)
    src, dst = (s, g) if src_is_smem else (g, s)

    def copy_tma(src_idx, dst_idx, **new_kwargs):
        cute.copy(atom, src[None, src_idx], dst[None, dst_idx], **new_kwargs, **kwargs)

    def copy_tma_single_stage(**new_kwargs):
        cute.copy(atom, src, dst, **new_kwargs, **kwargs)

    return (copy_tma if const_expr(not single_stage) else copy_tma_single_stage), s, g



class VarlenManager:
    bytes_per_tensormap = 128

    @dataclass
    class Params(ParamsBase):
        cu_seqlens_m: Optional[cute.Tensor] = None
        cu_seqlens_k: Optional[cute.Tensor] = None
        tensormaps: Optional[cute.Tensor] = None
        mAIdx: Optional[cute.Tensor] = None

        @staticmethod
        @cute.jit
        def create(args: VarlenArguments, *, loc=None, ip=None) -> "VarlenManager.Params":
            return VarlenManager.Params(
                cu_seqlens_m=args.mCuSeqlensM,
                cu_seqlens_k=args.mCuSeqlensK,
                tensormaps=args.mTensormaps,
                mAIdx=args.mAIdx,
            )

    def __init__(
        self,
        params: Params,
        tensormap_manager: Optional[cutlass.utils.TensorMapManager],
        tensormap_a_ptr: Optional[cute.Pointer],
        tensormap_b_ptr: Optional[cute.Pointer],
        tensormap_d_ptr: Optional[cute.Pointer],
        tensormap_epi_ptrs: list[Optional[cute.Pointer]],
        len_m_static: Int32,
        len_k_static: Int32,
        last_batch_idx: Int32 = Int32(-1),
        is_group_changed: Boolean = Boolean(True),
        *,
        loc=None,
        ip=None,
    ):
        self.params = params
        self.tensormap_manager = tensormap_manager
        self._tensormap_a_ptr = tensormap_a_ptr
        self._tensormap_b_ptr = tensormap_b_ptr
        self._tensormap_d_ptr = tensormap_d_ptr
        self._tensormap_epi_ptrs = tensormap_epi_ptrs
        self._len_m_static = len_m_static
        self._len_k_static = len_k_static
        self._last_batch_idx = last_batch_idx
        self._is_group_changed = is_group_changed
        self.varlen_m = const_expr(params.cu_seqlens_m is not None)
        self.varlen_k = const_expr(params.cu_seqlens_k is not None)
        self.gather_A = const_expr(params.mAIdx is not None)
        self._loc = loc
        self._ip = ip

    @staticmethod
    def to_underlying_arguments(args: VarlenArguments, *, loc=None, ip=None) -> Params:
        assert not (args.mCuSeqlensM is not None and args.mCuSeqlensK is not None), (
            "Only support either varlen_m or varlen_k"
        )
        return VarlenManager.Params.create(args, loc=loc, ip=ip)

    @staticmethod
    @cute.jit
    def create(
        params: Params,
        has_D: bool,
        num_epi_tensormaps: int,
        len_m_static: Int32,
        len_k_static: Int32,
        pingpong: bool = False,
        warp_idx: int | Int32 = 0,
        *,
        loc=None,
        ip=None,
    ) -> "VarlenManager":
        tensormap_manager = None
        tensormap_a_ptr, tensormap_b_ptr, tensormap_d_ptr = None, None, None
        tensormap_epi_ptrs = [None] * num_epi_tensormaps
        varlen_m = const_expr(params.cu_seqlens_m is not None)
        varlen_k = const_expr(params.cu_seqlens_k is not None)
        if const_expr(varlen_m or varlen_k):
            tensormap_manager = TensorMapManagerSm90(
                cutlass.utils.TensorMapUpdateMode.GMEM, VarlenManager.bytes_per_tensormap
            )
            # equivalent to bidx + bidy * gridDim.x + bidxz * gridDim.x * gridDim.y
            tensormap_workspace_idx = cute.make_layout(cute.arch.grid_dim())(cute.arch.block_idx())
            if const_expr(varlen_m):
                tensormap_d_idx = warp_idx // 4 if const_expr(pingpong) else 0
                tensormap_epi_offset = tensormap_d_idx
                if const_expr(has_D):
                    tensormap_d_ptr = tensormap_manager.get_tensormap_ptr(
                        params.tensormaps[tensormap_workspace_idx, tensormap_d_idx, None].iterator
                    )
                    tensormap_epi_offset += 1 if not pingpong else 2
                tensormap_epi_ptrs = [
                    tensormap_manager.get_tensormap_ptr(
                        params.tensormaps[
                            tensormap_workspace_idx,
                            tensormap_epi_offset + i * (1 if not pingpong else 2),
                            None,
                        ].iterator
                    )
                    for i in range(num_epi_tensormaps)
                ]
            else:
                assert varlen_k
                gather_A = const_expr(params.mAIdx is not None)
                if const_expr(not gather_A):
                    tensormap_a_ptr = tensormap_manager.get_tensormap_ptr(
                        params.tensormaps[tensormap_workspace_idx, 0, None].iterator
                    )
                tensormap_b_ptr = tensormap_manager.get_tensormap_ptr(
                    params.tensormaps[
                        tensormap_workspace_idx, 1 if not gather_A else 0, None
                    ].iterator
                )
        return VarlenManager(
            params,
            tensormap_manager,
            tensormap_a_ptr,
            tensormap_b_ptr,
            tensormap_d_ptr,
            tensormap_epi_ptrs,
            len_m_static=len_m_static,
            len_k_static=len_k_static,
        )

    def len_m(self, batch_idx: Int32) -> Int32:
        if const_expr(self.varlen_m):
            return self.params.cu_seqlens_m[batch_idx + 1] - self.params.cu_seqlens_m[batch_idx]
        else:
            return self._len_m_static

    def len_k(self, batch_idx: Int32) -> Int32:
        if const_expr(self.varlen_k):
            return self.params.cu_seqlens_k[batch_idx + 1] - self.params.cu_seqlens_k[batch_idx]
        else:
            return self._len_k_static

    def offset_batch_A(self, mA_mkl: cute.Tensor, batch_idx: Int32) -> cute.Tensor:
        params = self.params
        if const_expr(self.varlen_m):
            mA_mk = cute.domain_offset((params.cu_seqlens_m[batch_idx], 0), mA_mkl)
        elif const_expr(self.varlen_k):
            mA_mk = cute.domain_offset((0, params.cu_seqlens_k[batch_idx]), mA_mkl)
        else:
            mA_mk = mA_mkl[None, None, batch_idx]
        return mA_mk

    def offset_batch_AIdx(self, batch_idx: Int32) -> cute.Tensor:
        params = self.params
        if const_expr(self.varlen_m):
            mAIdx_mk = cute.domain_offset((params.cu_seqlens_m[batch_idx],), params.mAIdx)
        elif const_expr(self.varlen_k):
            mAIdx_mk = cute.domain_offset((params.cu_seqlens_k[batch_idx],), params.mAIdx)
        else:
            mAIdx_mk = params.mAIdx[None, batch_idx]
        return mAIdx_mk

    def offset_batch_B(self, mB_nkl: cute.Tensor, batch_idx: Int32) -> cute.Tensor:
        params = self.params
        if const_expr(self.varlen_k):
            mB_nk = cute.domain_offset((0, params.cu_seqlens_k[batch_idx]), mB_nkl)
        else:
            mB_nk = mB_nkl[None, None, batch_idx]
        return mB_nk

    def offset_batch_epi(self, mD_mnl: cute.Tensor, batch_idx: Int32) -> cute.Tensor:
        params = self.params
        if const_expr(self.varlen_m):
            mD_mn = cute.domain_offset((params.cu_seqlens_m[batch_idx], 0), mD_mnl)
        else:
            mD_mn = mD_mnl[None, None, batch_idx]
        return mD_mn

    def init_tensormap_AB(
        self,
        tma_atom_a: Optional[cute.CopyAtom],
        tma_atom_b: cute.CopyAtom,
        is_manager_warp: bool | Boolean = True,
    ) -> None:
        if const_expr(self.varlen_k):
            if const_expr(not self.gather_A):
                self.tensormap_manager.init_tensormap_from_atom(
                    tma_atom_a, self._tensormap_a_ptr, is_manager_warp
                )
            self.tensormap_manager.init_tensormap_from_atom(
                tma_atom_b, self._tensormap_b_ptr, is_manager_warp
            )

    def init_tensormap_epi(
        self,
        tma_atom_d: Optional[cute.CopyAtom],
        tma_atoms_epi: list[cute.CopyAtom],
        is_manager_warp: bool | Boolean = True,
    ) -> None:
        if const_expr(self.varlen_m):
            if const_expr(self._tensormap_d_ptr is not None):
                self.tensormap_manager.init_tensormap_from_atom(
                    tma_atom_d, self._tensormap_d_ptr, is_manager_warp
                )
            for tma_atom, tensormap_epi_ptr in zip(tma_atoms_epi, self._tensormap_epi_ptrs):
                self.tensormap_manager.init_tensormap_from_atom(
                    tma_atom, tensormap_epi_ptr, is_manager_warp
                )

    def fence_tensormap_init(self) -> None:
        self.tensormap_manager.fence_tensormap_initialization()

    @cute.jit
    def update_tensormap_AB(
        self,
        batch_idx: Int32,
        a_layout: LayoutEnum,
        b_layout: LayoutEnum,
        is_manager_warp: bool | Boolean = True,
    ) -> None:
        if const_expr(self.varlen_k):
            self._is_group_changed = Boolean(batch_idx != self._last_batch_idx)
            self._last_batch_idx = batch_idx
            if self._is_group_changed:
                # construct tensor A/B based on real address, shape and stride information
                cu_seqlens_k = self.params.cu_seqlens_k
                tensormap_ptrs = [self._tensormap_b_ptr]
                shapes = [cu_seqlens_k[batch_idx + 1]]
                orders = [0 if const_expr(b_layout == LayoutEnum.ROW_MAJOR) else 1]
                if const_expr(not self.gather_A):
                    tensormap_ptrs.insert(0, self._tensormap_a_ptr)
                    shapes.insert(0, cu_seqlens_k[batch_idx + 1])
                    orders.insert(0, 0 if const_expr(a_layout == LayoutEnum.ROW_MAJOR) else 1)
                self.tensormap_manager.update_tensormap_shape(
                    tensormap_ptrs,
                    is_manager_warp=is_manager_warp,
                    shapes=shapes,
                    orders=orders,
                    tensormap_smem_ptr=None,
                )

    @cute.jit
    def update_tensormap_epi(
        self,
        batch_idx: Int32,
        d_layout: LayoutEnum,
        epi_shapes: list[Int32],
        epi_orders: list[int],
        is_manager_warp: bool | Boolean = True,
    ) -> None:
        if const_expr(self.varlen_m):
            self._is_group_changed = Boolean(batch_idx != self._last_batch_idx)
            self._last_batch_idx = batch_idx
            # Cute-DSL doesn't like this under if statement
            order_d = (
                (0 if const_expr(d_layout.is_m_major_c()) else 1) if d_layout is not None else None
            )
            if self._is_group_changed:
                # construct tensor A/B based on real address, shape and stride information
                cu_seqlens_m = self.params.cu_seqlens_m
                # construct tensor D based on real address, shape and stride information
                tensormap_ptrs, shapes, orders = [], [], []
                if const_expr(self._tensormap_d_ptr is not None):
                    tensormap_ptrs.append(self._tensormap_d_ptr)
                    shapes.append(cu_seqlens_m[batch_idx + 1])
                    orders.append(order_d)
                tensormap_ptrs.extend(self._tensormap_epi_ptrs)
                shapes.extend(epi_shapes)
                orders.extend(epi_orders)
                self.tensormap_manager.update_tensormap_shape(
                    tensormap_ptrs,
                    is_manager_warp=is_manager_warp,
                    shapes=shapes,
                    orders=orders,
                    tensormap_smem_ptr=None,
                )

    @cute.jit
    def fence_tensormap_update_AB(self, is_manager_warp: bool | Boolean = True) -> None:
        if const_expr(self.varlen_k):
            if self._is_group_changed and is_manager_warp:
                if const_expr(not self.gather_A):
                    self.tensormap_manager.fence_tensormap_update(self._tensormap_a_ptr)
                self.tensormap_manager.fence_tensormap_update(self._tensormap_b_ptr)

    @cute.jit
    def fence_tensormap_update_epi(self, is_manager_warp: bool | Boolean = True) -> None:
        if const_expr(self.varlen_m):
            if self._is_group_changed and is_manager_warp:
                if const_expr(self._tensormap_d_ptr is not None):
                    self.tensormap_manager.fence_tensormap_update(self._tensormap_d_ptr)
                for tensormap_epi_ptr in self._tensormap_epi_ptrs:
                    if const_expr(tensormap_epi_ptr is not None):
                        self.tensormap_manager.fence_tensormap_update(tensormap_epi_ptr)

    def get_tma_desc_a_ptr(self) -> Optional[cute.Pointer]:
        tma_desc_a_ptr = None
        if const_expr(self.varlen_k and self._tensormap_a_ptr is not None):
            tma_desc_a_ptr = self.tensormap_manager.get_tensormap_ptr(
                self._tensormap_a_ptr, cute.AddressSpace.generic
            )
        return tma_desc_a_ptr

    def get_tma_desc_b_ptr(self) -> Optional[cute.Pointer]:
        tma_desc_b_ptr = None
        if const_expr(self.varlen_k):
            tma_desc_b_ptr = self.tensormap_manager.get_tensormap_ptr(
                self._tensormap_b_ptr, cute.AddressSpace.generic
            )
        return tma_desc_b_ptr

    def get_tma_desc_d_ptr(self) -> Optional[cute.Pointer]:
        tma_desc_d_ptr = None
        if const_expr(self.varlen_m and self._tensormap_d_ptr is not None):
            tma_desc_d_ptr = self.tensormap_manager.get_tensormap_ptr(
                self._tensormap_d_ptr, cute.AddressSpace.generic
            )
        return tma_desc_d_ptr

    def get_tma_desc_epi_ptrs(self) -> list[Optional[cute.Pointer]]:
        tma_desc_epi_ptrs = [None] * len(self._tensormap_epi_ptrs)
        if const_expr(self.varlen_m):
            for i, tensormap_epi_ptr in enumerate(self._tensormap_epi_ptrs):
                if const_expr(tensormap_epi_ptr is not None):
                    tma_desc_epi_ptrs[i] = self.tensormap_manager.get_tensormap_ptr(
                        tensormap_epi_ptr, cute.AddressSpace.generic
                    )
        return tma_desc_epi_ptrs

    def __extract_mlir_values__(self):
        values, self._values_pos = [], []
        for obj in [
            self.params,
            self.tensormap_manager,
            self._tensormap_a_ptr,
            self._tensormap_b_ptr,
            self._tensormap_d_ptr,
            self._tensormap_epi_ptrs,
            self._len_m_static,
            self._len_k_static,
            self._last_batch_idx,
            self._is_group_changed,
        ]:
            obj_values = cutlass.extract_mlir_values(obj)
            values += obj_values
            self._values_pos.append(len(obj_values))
        return values

    def __new_from_mlir_values__(self, values):
        obj_list = []
        for obj, n_items in zip(
            [
                self.params,
                self.tensormap_manager,
                self._tensormap_a_ptr,
                self._tensormap_b_ptr,
                self._tensormap_d_ptr,
                self._tensormap_epi_ptrs,
                self._len_m_static,
                self._len_k_static,
                self._last_batch_idx,
                self._is_group_changed,
            ],
            self._values_pos,
        ):
            obj_list.append(cutlass.new_from_mlir_values(obj, values[:n_items]))
            values = values[n_items:]
        return self.__class__(*(tuple(obj_list)), loc=self._loc)

@dataclass
class TileSchedulerArguments(ArgumentsBase):
    problem_shape_ntile_mnl: cute.Shape
    raster_order: cutlass.Constexpr[RasterOrderOption]
    group_size: Int32
    cluster_shape_mnk: cutlass.Constexpr[cute.Shape]
    tile_count_semaphore: Optional[cute.Pointer] = None
    batch_idx_permute: Optional[cute.Tensor] = None
    is_persistent: cutlass.Constexpr[bool] = False



# Grouping arguments together that should be passed to __call__
@dataclass
class TileSchedulerOptions(ArgumentsBase):
    max_active_clusters: Int32
    raster_order: cutlass.Constexpr[RasterOrderOption] = RasterOrderOption.Heuristic
    max_swizzle_size: Int32 = Int32(8)
    tile_count_semaphore: Optional[cute.Pointer] = None
    batch_idx_permute: Optional[cute.Tensor] = None



class PipelineStateWAdvance(PipelineState):
    def advance_iters(self, num_iterations: Int32):
        self._count += Int32(num_iterations)
        new_index = self._index + Int32(num_iterations)
        # How many times did we cross the stages boundary
        num_crossings = new_index // self.stages
        self._phase ^= num_crossings
        self._index = new_index % self.stages

    # This can be overridden by derived classes
    def __new_from_mlir_values__(self, values):
        return PipelineStateWAdvance(
            self.stages, Int32(values[0]), Int32(values[1]), Int32(values[2])
        )

@dataclass
class FastDivmod(ParamsBase):
    divisor: Int32
    multiplier: Uint32
    shift_right: Uint32

    # called by host
    @staticmethod
    def create(divisor: Int32) -> "FastDivmod":
        """Construct the FastDivmod object, in host code.
        This precomputes some values based on the divisor and is computationally expensive.
        """
        p = Uint32(31 + find_log2(divisor))
        divisor_u32 = Uint32(divisor)
        multiplier = Uint32(((cutlass.Uint64(1) << p) + divisor_u32 - 1) // divisor_u32)
        shift_right = Uint32(p - 32)
        return FastDivmod(divisor, multiplier, shift_right)

    @cute.jit
    def div(self, dividend: Int32) -> Int32:
        return (
            Int32(umulhi(dividend, self.multiplier) >> self.shift_right)
            if self.divisor != 1
            else dividend
        )

    def divmod(self, dividend: Int32) -> Tuple[Int32, Int32]:
        quotient = self.div(dividend)
        remainder = dividend - quotient * self.divisor
        return quotient, remainder




@cute.jit
def get_raster_order_from_option(
    raster_order_option: RasterOrderOption, problem_shape_ncluster_mn: cute.Shape, group_size: Int32
) -> RasterOrder:
    raster_order = (
        RasterOrder.AlongM
        if raster_order_option == RasterOrderOption.AlongM
        else RasterOrder.AlongN
    )
    if raster_order_option == RasterOrderOption.Heuristic:
        problem_blocks_m = cute.round_up(problem_shape_ncluster_mn[0], group_size)
        problem_blocks_n = cute.round_up(problem_shape_ncluster_mn[1], group_size)
        raster_order = (
            RasterOrder.AlongM if problem_blocks_n > problem_blocks_m else RasterOrder.AlongN
        )
    return raster_order


class TileScheduler:
    @dataclass
    class Params(ParamsBase):
        problem_shape_ncluster_mnl: cute.Shape
        raster_order: RasterOrder
        num_clusters_per_problem_divmod: FastDivmod
        num_groups_regular: Int32
        group_size_divmod: FastDivmod
        group_size_tail_divmod: FastDivmod
        num_clusters_in_group_divmod: FastDivmod
        tile_count_semaphore: Optional[cute.Pointer]
        batch_idx_permute: Optional[cute.Tensor]
        cluster_shape_mn: cutlass.Constexpr[cute.Shape]
        is_persistent: cutlass.Constexpr[bool]

        @staticmethod
        @cute.jit
        def create(args: TileSchedulerArguments, *, loc=None, ip=None) -> "TileScheduler.Params":
            assert args.cluster_shape_mnk[2] == 1
            cluster_shape_mn = const_expr(cute.select(args.cluster_shape_mnk, mode=[0, 1]))
            problem_shape_ntile_mn = cute.select(args.problem_shape_ntile_mnl, mode=[0, 1])
            problem_shape_ncluster_mn = cute.ceil_div(problem_shape_ntile_mn, cluster_shape_mn)
            problem_shape_ncluster_mnl = problem_shape_ncluster_mn + (
                args.problem_shape_ntile_mnl[2],
            )
            num_clusters_per_problem = cute.size(problem_shape_ncluster_mn)
            raster_order = get_raster_order_from_option(
                args.raster_order, problem_shape_ncluster_mn, args.group_size
            )
            ncluster_fast = (
                problem_shape_ncluster_mn[0]
                if raster_order == RasterOrder.AlongM
                else problem_shape_ncluster_mn[1]
            )
            ncluster_slow = (
                problem_shape_ncluster_mn[1]
                if raster_order == RasterOrder.AlongM
                else problem_shape_ncluster_mn[0]
            )
            group_size = min(args.group_size, ncluster_fast)
            group_size_tail = ncluster_fast % group_size
            num_groups_regular = ncluster_fast // group_size
            num_clusters_in_group = group_size * ncluster_slow
            return TileScheduler.Params(
                problem_shape_ncluster_mnl,
                raster_order,
                FastDivmod.create(num_clusters_per_problem),
                num_groups_regular,
                FastDivmod.create(group_size),
                # Don't divide by 0
                FastDivmod.create(group_size_tail if group_size_tail > 0 else 1),
                FastDivmod.create(num_clusters_in_group),
                args.tile_count_semaphore if const_expr(args.is_persistent) else None,
                args.batch_idx_permute,
                cluster_shape_mn,
                args.is_persistent,
            )

    def __init__(
        self,
        current_work_linear_idx: Int32,
        num_tiles_executed: Int32,
        tile_count: Optional[cute.Tensor],
        scheduler_pipeline: Optional[cutlass.pipeline.PipelineAsync],
        pipeline_state: PipelineStateWAdvance,
        params: Params,
        *,
        loc=None,
        ip=None,
    ):
        self._current_work_linear_idx = current_work_linear_idx
        self.num_tiles_executed = num_tiles_executed
        self._tile_count = tile_count
        self._scheduler_pipeline = scheduler_pipeline
        self._pipeline_state = pipeline_state
        self.params = params
        self._loc = loc
        self._ip = ip

    @staticmethod
    def to_underlying_arguments(args: TileSchedulerArguments, *, loc=None, ip=None) -> Params:
        return TileScheduler.Params.create(args, loc=loc, ip=ip)

    @staticmethod
    @cute.jit
    def create(
        params: Params,
        tile_count: Optional[cute.Tensor] = None,
        scheduler_pipeline: Optional[cutlass.pipeline.PipelineAsync] = None,
        is_scheduler_warp: bool | Boolean = False,
        *,
        loc=None,
        ip=None,
    ) -> "TileScheduler":
        """is_scheduler_warp should only be true for one warp in the whole cluster"""
        stages = 0
        if const_expr(not params.is_persistent):
            cidx, cidy, _ = cute.arch.cluster_idx()
            cdimx, _, _ = cute.arch.cluster_dim()
            cluster_id = cidx + cidy * cdimx
            current_work_linear_idx = Int32(cluster_id)
        else:
            _, _, bidz = cute.arch.block_idx()
            current_work_linear_idx = Int32(bidz)
            if const_expr(params.tile_count_semaphore is not None):
                assert tile_count is not None
                assert scheduler_pipeline is not None
                stages = const_expr(cute.size(tile_count))
        return TileScheduler(
            current_work_linear_idx,
            Int32(0),  # num_tiles_executed
            tile_count,
            scheduler_pipeline,
            PipelineStateWAdvance(stages, Int32(0), Int32(0), Int32(1 if is_scheduler_warp else 0)),
            params,
            loc=loc,
            ip=ip,
        )

    # called by host
    @staticmethod
    def get_grid_shape(
        params: Params,
        max_active_clusters: Int32,
        *,
        loc=None,
        ip=None,
    ) -> Tuple[Int32, Int32, Int32]:
        num_ctas_mnl = tuple(
            x * y for x, y in zip(params.problem_shape_ncluster_mnl, params.cluster_shape_mn)
        ) + (params.problem_shape_ncluster_mnl[2],)
        if const_expr(not params.is_persistent):
            return num_ctas_mnl
        else:
            num_ctas_in_problem = cute.size(num_ctas_mnl, loc=loc, ip=ip)
            num_ctas_per_cluster = cute.size(params.cluster_shape_mn, loc=loc, ip=ip)
            # Total ctas that can run in one wave
            num_ctas_per_wave = max_active_clusters * num_ctas_per_cluster
            num_persistent_ctas = cutlass.min(num_ctas_in_problem, num_ctas_per_wave)
            num_persistent_clusters = num_persistent_ctas // num_ctas_per_cluster
            return (*params.cluster_shape_mn, num_persistent_clusters)

    @cute.jit
    def _swizzle_cta(
        self, cluster_id_in_problem: Int32, *, loc=None, ip=None
    ) -> Tuple[Int32, Int32]:
        # CTA Swizzle to promote L2 data reuse
        params = self.params
        group_id, id_in_group = params.num_clusters_in_group_divmod.divmod(cluster_id_in_problem)
        cid_fast_in_group, cid_slow = Int32(0), Int32(0)
        if group_id < params.num_groups_regular:
            cid_slow, cid_fast_in_group = params.group_size_divmod.divmod(id_in_group)
        else:  # tail part
            cid_slow, cid_fast_in_group = params.group_size_tail_divmod.divmod(id_in_group)
        if group_id % 2 == 1:  # serpentine order
            ncluster_slow = (
                params.problem_shape_ncluster_mnl[1]
                if params.raster_order == RasterOrder.AlongM
                else params.problem_shape_ncluster_mnl[0]
            )
            cid_slow = ncluster_slow - 1 - cid_slow
        cid_fast = group_id * params.group_size_divmod.divisor + cid_fast_in_group
        cid_m, cid_n = cid_fast, cid_slow
        if params.raster_order == RasterOrder.AlongN:
            cid_m, cid_n = cid_slow, cid_fast
        return cid_m, cid_n

    @cute.jit
    def get_current_work(self, *, loc=None, ip=None) -> cutlass.utils.WorkTileInfo:
        params = self.params
        if const_expr(not params.is_persistent):
            cluster_id_in_problem = self._current_work_linear_idx
            _, _, bidz = cute.arch.block_idx()
        else:
            bidz, cluster_id_in_problem = params.num_clusters_per_problem_divmod.divmod(
                self._current_work_linear_idx
            )
        cid_m, cid_n = self._swizzle_cta(cluster_id_in_problem, loc=loc, ip=ip)
        # Get the pid from cluster id
        bidx_in_cluster = cute.arch.block_in_cluster_idx()
        pid_m = cid_m * params.cluster_shape_mn[0] + bidx_in_cluster[0]
        pid_n = cid_n * params.cluster_shape_mn[1] + bidx_in_cluster[1]
        batch_idx = (
            bidz if const_expr(params.batch_idx_permute is None) else params.batch_idx_permute[bidz]
        )
        tile_coord_mnkl = (pid_m, pid_n, None, batch_idx)
        if const_expr(not params.is_persistent):
            is_valid = self.num_tiles_executed == 0
        else:
            is_valid = self._current_work_linear_idx < cute.size(params.problem_shape_ncluster_mnl)
        return cutlass.utils.WorkTileInfo(tile_coord_mnkl, is_valid)

    def initial_work_tile_info(self, *, loc=None, ip=None):
        return self.get_current_work(loc=loc, ip=ip)

    @cute.jit
    def fetch_next_work(self, is_scheduler_warp: bool | Boolean = False, *, loc=None, ip=None):
        """is_scheduler_warp should only be true for one warp in the whole cluster"""
        params = self.params
        if const_expr(params.is_persistent and params.tile_count_semaphore is not None):
            current_work_linear_idx = self._current_work_linear_idx
            if is_scheduler_warp:
                if cute.arch.lane_idx() == 0:
                    num_persistent_clusters = cute.arch.grid_dim()[2]
                    current_work_linear_idx = num_persistent_clusters + utils.atomic_inc_i32(
                        cute.size(params.problem_shape_ncluster_mnl) - 1,
                        params.tile_count_semaphore,
                    )
                # lane 0 already has the right tile_idx, just need to broadcast
                current_work_linear_idx = cute.arch.shuffle_sync(current_work_linear_idx, 0)
            self._current_work_linear_idx = current_work_linear_idx

    @cute.jit
    def advance_to_next_work(
        self,
        is_scheduler_warp: bool | Boolean = False,
        *,
        advance_count: int = 1,
        loc=None,
        ip=None,
    ):
        tidx = cute.arch.thread_idx()[0]
        bidx = cute.arch.block_idx()[0]
        bidz = cute.arch.block_idx()[2]
        params = self.params
        if const_expr(params.is_persistent):
            num_persistent_clusters = cute.arch.grid_dim()[2]
            if const_expr(params.tile_count_semaphore is None):  # Static persistent
                self._current_work_linear_idx += advance_count * Int32(num_persistent_clusters)
            else:  # Dynamic persistent
                if const_expr(advance_count > 1):
                    self._pipeline_state.advance_iters(advance_count - 1)
                current_work_linear_idx = self._current_work_linear_idx
                if is_scheduler_warp:
                    self._scheduler_pipeline.producer_acquire(self._pipeline_state)
                    lane_idx = cute.arch.lane_idx()
                    if lane_idx < cute.size(params.cluster_shape_mn):
                        # cute.printf("Producer bidx = {}, bidz = {}, tidx = {}, after empty wait, idx = {}", bidx, bidz, tidx, current_work_linear_idx)
                        if const_expr(cute.size(params.cluster_shape_mn) == 1):
                            self._tile_count[self._pipeline_state.index] = current_work_linear_idx
                            self._scheduler_pipeline.producer_commit(self._pipeline_state)
                        else:
                            peer_cta_rank_in_cluster = lane_idx
                            mbar_ptr = self._scheduler_pipeline.producer_get_barrier(
                                self._pipeline_state
                            )
                            cute.arch.mbarrier_arrive_and_expect_tx(
                                mbar_ptr, 4, peer_cta_rank_in_cluster
                            )
                            utils.store_shared_remote(
                                val=current_work_linear_idx,
                                smem_ptr=self._tile_count.iterator + self._pipeline_state.index,
                                mbar_ptr=mbar_ptr,
                                peer_cta_rank_in_cluster=peer_cta_rank_in_cluster,
                            )
                        # cute.printf("Producer bidx = {}, bidz = {}, tidx = {}, after full arrive", bidx, bidz, tidx)
                else:
                    # if tidx % 32 == 0: cute.printf("bidx = {}, bidz = {}, tidx = {}, before full wait, idx = {}", bidx, bidz, tidx, current_work_linear_idx)
                    self._scheduler_pipeline.consumer_wait(self._pipeline_state)
                    # if tidx % 32 == 0: cute.printf("bidx = {}, bidz = {}, tidx = {}, after full wait, idx = {}", bidx, bidz, tidx, current_work_linear_idx)
                    current_work_linear_idx = self._tile_count[self._pipeline_state.index]
                    # if tidx % 32 == 0: cute.printf("bidx = {}, bidz = {}, tidx = {}, after smem read, idx = {}", bidx, bidz, tidx, current_work_linear_idx)
                    # Need this fence since the STAS from the producer is using the async proxy.
                    # Without this, we get race condition / deadlock.
                    if const_expr(cute.size(params.cluster_shape_mn) > 1):
                        cute.arch.fence_proxy(
                            cute.arch.ProxyKind.async_shared, space=cute.arch.SharedSpace.shared_cta
                        )
                    cute.arch.sync_warp()
                    with cute.arch.elect_one():
                        # if tidx % 32 == 0: cute.printf("bidx = {}, bidz = {}, tidx = {}, before empty arrive", bidx, bidz, tidx)
                        self._scheduler_pipeline.consumer_release(self._pipeline_state)
                        # if tidx == 320: cute.printf("bidx = {}, bidz = {}, tidx = {}, idx = {}, after empty arrive", bidx, bidz, tidx, current_work_linear_idx)
                    # if tidx == 320: cute.printf("bidx = {}, bidz = {}, tidx = {}, idx = {}, after empty arrive", bidx, bidz, tidx, current_work_linear_idx)
                self._current_work_linear_idx = current_work_linear_idx
                self._pipeline_state.advance()
        self.num_tiles_executed += Int32(advance_count)

    def producer_tail(self):
        if const_expr(self.params.is_persistent and self.params.tile_count_semaphore is not None):
            self._scheduler_pipeline.producer_tail(self._pipeline_state)

    def __extract_mlir_values__(self):
        values, self._values_pos = [], []
        for obj in [
            self._current_work_linear_idx,
            self.num_tiles_executed,
            self._tile_count,
            self._scheduler_pipeline,
            self._pipeline_state,
            self.params,
        ]:
            obj_values = cutlass.extract_mlir_values(obj)
            values += obj_values
            self._values_pos.append(len(obj_values))
        return values

    def __new_from_mlir_values__(self, values):
        obj_list = []
        for obj, n_items in zip(
            [
                self._current_work_linear_idx,
                self.num_tiles_executed,
                self._tile_count,
                self._scheduler_pipeline,
                self._pipeline_state,
                self.params,
            ],
            self._values_pos,
        ):
            obj_list.append(cutlass.new_from_mlir_values(obj, values[:n_items]))
            values = values[n_items:]
        return self.__class__(*(tuple(obj_list)), loc=self._loc)



# return PipelineStateWAdvance instead of PipelineState

class GemmSm90:
    """
    This class implements batched matrix multiplication (C = A x B) with support for various data types
    and architectural features specific to Hopper GPUs with persistent tile scheduling and warp specialization.

    :param acc_dtype: Data type for accumulation during computation
    :type acc_dtype: type[cutlass.Numeric]
    :param tile_shape_mn: Shape of the CTA tile (M,N)
    :type tile_shape_mn: Tuple[int, int, int]
    :param cluster_shape_mnk: Cluster dimensions (M,N,K) for parallel processing
    :type cluster_shape_mnk: Tuple[int, int, int]

    :note: Data type requirements:
        - For 16-bit types: A and B must have the same data type
        - For 8-bit types: A and B can have different types (Float8E4M3FN/Float8E5M2) as long as both are 8-bit
        - Float8 types only support k-major layout

    :note: Supported data types:
        - Float16
        - BFloat16
        - Float8E4M3FN/Float8E5M2

    :note: Supported accumulation types:
        - Float32 (for all floating point inputs)

    :note: Constraints:
        - Cluster shape M/N must be positive and power of 2, total cluster size <= 4

    Example:
        >>> gemm = GemmSm90(
        ...     acc_dtype=Float32,
        ...     tile_shape_mn=(128, 256),
        ...     cluster_shape_mnk=(1, 1, 1)
        ... )
        >>> gemm(a_tensor, b_tensor, c_tensor, stream)
    """

    arch = 90
    num_epi_tensormaps: int = 0

    EpilogueArguments = ArgumentsBase
    EpilogueParams = ParamsBase

    def __init__(
        self,
        acc_dtype: Type[cutlass.Numeric],
        a_dtype: Type[cutlass.Numeric],
        tile_shape_mn: Tuple[int, int],
        cluster_shape_mnk: Tuple[int, int, int],
        pingpong: bool = False,
        is_persistent: bool = True,
        fp8_fast_accum: bool = False,
        gather_A: bool = False,
    ):
        """
        Initializes the configuration for a Hopper dense GEMM kernel.

        This configuration includes data types for operands, tile shape, cluster configuration,
        and thread layout.

        :param acc_dtype: Data type for accumulation during computation
        :type acc_dtype: type[cutlass.Numeric]
        :param tile_shape_mn: Shape of the CTA tile (M,N)
        :type tile_shape_mn: Tuple[int, int]
        :param cluster_shape_mnk: Cluster dimensions (M,N,K) for parallel processing
        :type cluster_shape_mnk: Tuple[int, int, int]
        """

        self.acc_dtype = acc_dtype
        self.pingpong = pingpong
        self.is_persistent = is_persistent
        if self.pingpong:
            assert self.is_persistent, "Pingpong gemm requires persistent scheduler"
        self.fp8_slow_accum = not fp8_fast_accum and a_dtype.width == 8
        self.gather_A = gather_A
        if gather_A:
            assert cluster_shape_mnk[1] == 1, "Cluster shape N must be 1 for gather A "

        self.cluster_shape_mnk = cluster_shape_mnk
        # K dimension is deferred in _setup_attributes
        self.cta_tile_shape_mnk = (*tile_shape_mn, 1)
        tile_M, tile_N = self.cta_tile_shape_mnk[0], self.cta_tile_shape_mnk[1]
        # check the cta tile shape
        if not self.pingpong:
            if tile_M not in [64, 128, 192, 256, 320]:
                raise ValueError("CTA tile shape M must be 64/128/192/256/320")
            if tile_M in [192, 320]:  # special case
                tile_N_max = 256 if tile_M == 192 else 160
                if not (tile_N % 32 == 0 and tile_N <= tile_N_max):
                    raise ValueError(
                        f"If tile_m == {tile_M}, CTA tile shape N must be divisible by 32 and <= {tile_N_max}"
                    )
            else:
                if not (
                    (tile_N % 16 == 0 and tile_N <= 256) or (tile_N % 32 == 0 and tile_N <= 512)
                ):
                    raise ValueError(
                        "CTA tile shape N must be divisible by 16 and <= 256, or divisible by 32 and <= 512"
                    )
        else:
            if tile_M not in [64, 128, 192]:
                raise ValueError("CTA tile shape M must be 64/128/192 if pingpong")
            tile_N_max = 256 if tile_M == 64 else (208 if tile_M == 128 else 128)
            if not (tile_N % 16 == 0 and tile_N <= tile_N_max):
                raise ValueError(f"CTA tile shape N must be divisible by 16 and <= {tile_N_max}")

        if not self.pingpong:
            if tile_M == 320:  # tile_M / 64 is not even so we have to split along N
                atom_layout_m, atom_layout_n = 1, 2
            elif tile_M == 192:
                if tile_N <= 128:
                    atom_layout_m, atom_layout_n = 3, 1
                else:
                    atom_layout_m, atom_layout_n = 1, 2
            else:
                atom_layout_m = (
                    self.cta_tile_shape_mnk[0] // 64 if self.cta_tile_shape_mnk[0] < 256 else 2
                )
                atom_layout_n = 1
            assert atom_layout_m in [1, 2, 3] and atom_layout_n in [1, 2]
        else:
            atom_layout_m, atom_layout_n = 1, 1
        self.atom_layout_mnk = (atom_layout_m, atom_layout_n, 1)

        self.num_mcast_ctas_a = self.cluster_shape_mnk[1]
        if self.gather_A:
            assert self.num_mcast_ctas_a == 1
        self.num_mcast_ctas_b = self.cluster_shape_mnk[0]
        self.is_a_mcast = self.num_mcast_ctas_a > 1
        self.is_b_mcast = self.num_mcast_ctas_b > 1

        self.occupancy = 1
        self.mma_warp_groups = math.prod(self.atom_layout_mnk) * (1 if not self.pingpong else 2)
        if self.pingpong:
            assert self.mma_warp_groups == 2
        assert self.mma_warp_groups in [1, 2, 3]
        self.num_threads_per_warp_group = 128
        self.threads_per_cta = (self.mma_warp_groups + 1) * self.num_threads_per_warp_group
        self.smem_capacity = cutlass.utils.get_smem_capacity_in_bytes("sm_90")
        self.num_epi_warps = (self.mma_warp_groups if not self.pingpong else 1) * 4
        self.num_ab_load_warps = 1 if not self.gather_A else 4
        self.ab_load_warp_id = self.mma_warp_groups * 4
        # self.num_epi_load_threads = cute.arch.WARP_SIZE * 1
        # self.epi_load_warp_id = self.ab_load_warp_id + self.num_ab_load_warps

        regs_per_thread = math.prod(self.cta_tile_shape_mnk[:2]) // (
            math.prod(self.atom_layout_mnk) * self.num_threads_per_warp_group
        )
        if self.fp8_slow_accum:
            regs_per_thread *= 2
        if not self.gather_A:
            if self.mma_warp_groups == 3:
                self.num_regs_load, self.num_regs_mma = 32, 160
            else:
                heavy_register_pressure = regs_per_thread >= 208
                self.num_regs_load, self.num_regs_mma = (
                    (40, 232) if not heavy_register_pressure else (24, 240)
                )
        else:
            if self.mma_warp_groups == 3:
                self.num_regs_load, self.num_regs_mma = 56, 152
            else:
                self.num_regs_load, self.num_regs_mma = (56, 224)

        self.ab_stage = None
        self.epi_stage = None

        self.a_smem_layout_staged = None
        self.b_smem_layout_staged = None
        self.epi_smem_layout_staged = None
        self.epi_tile = None

        self.shared_storage = None
        self.buffer_align_bytes = 1024

    def _setup_attributes(self, epilogue_args: EpilogueArguments):
        """Set up configurations that are dependent on GEMM inputs

        This method configures various attributes based on the input tensor properties
        (data types, leading dimensions) and kernel settings:
        - Configuring tiled MMA
        - Computing MMA/cluster/tile shapes
        - Computing cluster layout
        - Computing multicast CTAs for A/B
        - Computing epilogue subtile
        - Setting up A/B/C stage counts in shared memory
        - Computing A/B/C shared memory layout
        """

        self.tiled_mma = sm90_utils.make_trivial_tiled_mma(
            self.a_dtype,
            self.b_dtype,
            self.a_layout.sm90_mma_major_mode(),
            self.b_layout.sm90_mma_major_mode(),
            self.acc_dtype,
            self.atom_layout_mnk,
            tiler_mn=(64, self.cta_tile_shape_mnk[1] // self.atom_layout_mnk[1]),
        )
        if const_expr(self.atom_layout_mnk[1] > 1):
            # If N dimension is split among 2 WGs, we need to permute the N dimension so
            # that in the epilogue, WG0 and WG1 can write to epi smem of size e.g. (64, 32)
            # containing accumulators that are next to each other in the N dimension.
            # Without permutation WG0 would write to epi smem of size (64, 16) and
            # WG1 would write to a separate epi smem of size (64, 16) that's far away.
            atom_n = self.atom_layout_mnk[1]
            permutation_n = cute.make_ordered_layout(
                (8, self.cta_tile_shape_mnk[1] // atom_n // 8, atom_n), order=(0, 2, 1)
            )
            self.tiled_mma = cute.make_tiled_mma(
                cute.make_mma_atom(self.tiled_mma.op),
                self.atom_layout_mnk,
                permutation_mnk=(None, permutation_n, None),
            )
        mma_inst_shape_k = cute.size(self.tiled_mma.shape_mnk, mode=[2])
        mma_inst_tile_k = 4
        self.cta_tile_shape_mnk = (
            self.cta_tile_shape_mnk[0],
            self.cta_tile_shape_mnk[1],
            mma_inst_shape_k * mma_inst_tile_k,
        )

        self.cluster_layout_mnk = cute.make_layout(self.cluster_shape_mnk)

        self.epi_tile = self._sm90_compute_tile_shape_or_override(
            self.cta_tile_shape_mnk,
            self.atom_layout_mnk,
            self.d_dtype,
        )

        # Compute stage before compute smem layout
        self.ab_stage, self.epi_stage, self.epi_c_stage = self._compute_stages(
            self.cta_tile_shape_mnk,
            self.epi_tile,
            self.a_dtype,
            self.b_dtype,
            self.d_dtype,
            self.c_dtype,
            epilogue_args,
            cutlass.utils.get_smem_capacity_in_bytes(f"sm_{self.arch}"),  # smem_capacity
            self.occupancy,
            # epi_smem will reuse smem ab if not persistent.
            overlap_sD_sA=not self.is_persistent,
        )
        self.sched_stage = 2 if self.pingpong else 1

        (
            self.a_smem_layout_staged,
            self.b_smem_layout_staged,
            self.epi_smem_layout_staged,
            self.epi_c_smem_layout_staged,
        ) = self._make_smem_layouts(
            self.cta_tile_shape_mnk,
            self.epi_tile,
            self.a_dtype,
            self.a_layout,
            self.b_dtype,
            self.b_layout,
            self.ab_stage,
            self.d_dtype,
            self.d_layout,
            self.epi_stage,
            self.c_dtype,
            self.c_layout,
            self.epi_c_stage,
        )

    @cute.jit
    def __call__(
        self,
        mA: cute.Tensor,
        mB: cute.Tensor,
        mD: Optional[cute.Tensor],
        mC: Optional[cute.Tensor],
        epilogue_args: ArgumentsBase,
        scheduler_args: TileSchedulerOptions,
        varlen_args: Optional[VarlenArguments],
        stream: cuda.CUstream,
    ):
        """Execute the GEMM operation in steps:
        - Setup static attributes
        - Setup TMA load/store atoms and tensors
        - Compute grid size
        - Define shared storage for kernel
        - Launch the kernel synchronously

        :param mA: Input tensor A
        :type mA: cute.Tensor
        :param mB: Input tensor B
        :type mB: cute.Tensor
        :param mD: Output tensor D
        :type mD: cute.Tensor
        :param stream: CUDA stream for asynchronous execution
        :type stream: cuda.CUstream
        """

        # setup static attributes before smem/grid/tma computation
        self.a_dtype = mA.element_type
        self.b_dtype = mB.element_type
        self.d_dtype = mD.element_type if mD is not None else None
        self.c_dtype = mC.element_type if mC is not None else None
        self.a_layout = LayoutEnum.from_tensor(mA)
        self.b_layout = LayoutEnum.from_tensor(mB)
        self.d_layout = LayoutEnum.from_tensor(mD) if mD is not None else None
        self.c_layout = LayoutEnum.from_tensor(mC) if mC is not None else None

        if const_expr(self.a_dtype.width == 16 and self.a_dtype != self.b_dtype):
            raise TypeError(f"Type mismatch: {self.a_dtype} != {self.b_dtype}")
        if const_expr(self.a_dtype.width != self.b_dtype.width):
            raise TypeError(f"Type width mismatch: {self.a_dtype.width} != {self.b_dtype.width}")
        if const_expr(self.a_dtype.width != 16 and self.a_dtype.width != 8):
            raise TypeError("a_dtype should be float16 or float8")

        if const_expr(varlen_args is None):
            varlen_args = VarlenArguments()
        assert (varlen_args.mAIdx is not None) == self.gather_A

        # Assume all strides are divisible by 128 bits except the last stride
        new_stride = lambda t: tuple(
            cute.assume(s, divby=128 // t.element_type.width) if not cute.is_static(s) else s
            for s in t.stride
        )
        mA, mD = [
            cute.make_tensor(t.iterator, cute.make_layout(t.shape, stride=new_stride(t)))
            if t is not None
            else None
            for t in (mA, mD)
        ]

        self._setup_attributes(epilogue_args)

        a_smem_layout = cute.slice_(self.a_smem_layout_staged, (None, None, 0))
        b_smem_layout = cute.slice_(self.b_smem_layout_staged, (None, None, 0))
        tma_atom_a, tma_tensor_a = None, None
        if const_expr(not self.gather_A):
            tma_atom_a, tma_tensor_a = self._make_tma_atoms_and_tensors(
                mA,
                a_smem_layout,
                (self.cta_tile_shape_mnk[0], self.cta_tile_shape_mnk[2]),
                self.cluster_shape_mnk[1],
            )
        tma_atom_b, tma_tensor_b = self._make_tma_atoms_and_tensors(
            mB,
            b_smem_layout,
            (self.cta_tile_shape_mnk[1], self.cta_tile_shape_mnk[2]),
            self.cluster_shape_mnk[0],
        )

        self.num_tma_load_bytes = cute.size_in_bytes(self.b_dtype, b_smem_layout)
        if const_expr(not self.gather_A):
            self.num_tma_load_bytes += cute.size_in_bytes(self.a_dtype, a_smem_layout)

        tma_atom_d, tma_tensor_d = None, None
        if const_expr(mD is not None):
            tma_atom_d, tma_tensor_d = self._make_tma_epi_atoms_and_tensors(
                mD,
                self.epi_smem_layout_staged,
                self.epi_tile,
                op_type="store"
                if not (hasattr(epilogue_args, "add_to_output") and epilogue_args.add_to_output)
                else "add",
            )
        tma_atom_c, tma_tensor_c = None, None
        if const_expr(mC is not None):
            tma_atom_c, tma_tensor_c = self._make_tma_epi_atoms_and_tensors(
                mC, self.epi_c_smem_layout_staged, self.epi_tile, op_type="load"
            )

        epilogue_params = self.epi_to_underlying_arguments(epilogue_args)
        varlen_params = VarlenManager.to_underlying_arguments(varlen_args)

        TileSchedulerCls = self.get_scheduler_class(varlen_m=varlen_args.mCuSeqlensM is not None)
        tile_sched_args = self.get_scheduler_arguments(mA, mB, mD, scheduler_args, varlen_args)
        tile_sched_params = TileSchedulerCls.to_underlying_arguments(tile_sched_args)
        grid = TileSchedulerCls.get_grid_shape(
            tile_sched_params, scheduler_args.max_active_clusters
        )

        epi_smem_size = (
            cute.cosize(self.epi_smem_layout_staged) if self.is_persistent and mD is not None else 0
        )
        epi_c_smem_size = cute.cosize(self.epi_c_smem_layout_staged) if mC is not None else 0

        @cute.struct
        class SharedStorage:
            ab_pipeline_array_ptr: cute.struct.MemRange[cutlass.Int64, self.ab_stage * 2]
            epi_pipeline_array_ptr: cute.struct.MemRange[cutlass.Int64, self.epi_c_stage * 2]
            sched_pipeline_array_ptr: cute.struct.MemRange[cutlass.Int64, self.sched_stage * 2]
            tile_count: cute.struct.MemRange[Int32, self.sched_stage]
            sD: cute.struct.Align[
                cute.struct.MemRange[
                    self.d_dtype if self.d_dtype is not None else Int32, epi_smem_size
                ],
                self.buffer_align_bytes,
            ]
            sC: cute.struct.Align[
                cute.struct.MemRange[
                    self.c_dtype if self.c_dtype is not None else Int32, epi_c_smem_size
                ],
                self.buffer_align_bytes,
            ]
            epi: self.epi_get_smem_struct(epilogue_params)
            sA: cute.struct.Align[
                cute.struct.MemRange[self.a_dtype, cute.cosize(self.a_smem_layout_staged)],
                self.buffer_align_bytes,
            ]
            sB: cute.struct.Align[
                cute.struct.MemRange[self.b_dtype, cute.cosize(self.b_smem_layout_staged)],
                self.buffer_align_bytes,
            ]

        self.shared_storage = SharedStorage

        # Launch the kernel synchronously
        self.kernel(
            self.tiled_mma,
            tma_atom_a,
            tma_tensor_a if const_expr(not self.gather_A) else mA,
            tma_atom_b,
            tma_tensor_b,
            tma_atom_d,
            tma_tensor_d,
            tma_atom_c,
            tma_tensor_c,
            epilogue_params,
            varlen_params,
            self.cluster_layout_mnk,
            self.a_smem_layout_staged,
            self.b_smem_layout_staged,
            self.epi_smem_layout_staged,
            self.epi_c_smem_layout_staged,
            tile_sched_params,
            TileSchedulerCls,
        ).launch(
            grid=grid,
            block=[self.threads_per_cta, 1, 1],
            cluster=self.cluster_shape_mnk,
            stream=stream,
            min_blocks_per_mp=1,
        )
        return

    #  GPU device kernel
    @cute.kernel
    def kernel(
        self,
        tiled_mma: cute.TiledMma,
        tma_atom_a: Optional[cute.CopyAtom],
        mA_mkl: cute.Tensor,
        tma_atom_b: cute.CopyAtom,
        mB_nkl: cute.Tensor,
        tma_atom_d: Optional[cute.CopyAtom],
        mD_mnl: Optional[cute.Tensor],
        tma_atom_c: Optional[cute.CopyAtom],
        mC_mnl: Optional[cute.Tensor],
        epilogue_params: ParamsBase,
        varlen_params: VarlenManager.Params,
        cluster_layout_mnk: cute.Layout,
        a_smem_layout: cute.ComposedLayout,
        b_smem_layout: cute.ComposedLayout,
        epi_smem_layout: cute.ComposedLayout,
        epi_c_smem_layout: cute.ComposedLayout,
        tile_sched_params: ParamsBase,
        TileSchedulerCls: cutlass.Constexpr[Callable],
    ):
        """
        GPU device kernel performing the batched GEMM computation.

        :param tma_atom_a: TMA copy atom for A tensor
        :type tma_atom_a: cute.CopyAtom
        :param mA_mkl: Input tensor A
        :type mA_mkl: cute.Tensor
        :param tma_atom_b: TMA copy atom for B tensor
        :type tma_atom_b: cute.CopyAtom
        :param mB_nkl: Input tensor B
        :type mB_nkl: cute.Tensor
        :param tma_atom_d: TMA copy atom for D tensor
        :type tma_atom_d: cute.CopyAtom
        :param mD_mnl: Output tensor D
        :type mD_mnl: cute.Tensor
        :param tiled_mma: Tiled MMA object
        :type tiled_mma: cute.TiledMma
        :param cluster_layout_mnk: CTA layout
        :type cluster_layout_mnk: cute.Layout
        :param a_smem_layout: Shared memory layout for A
        :type a_smem_layout: cute.ComposedLayout
        :param b_smem_layout: Shared memory layout for B
        :type b_smem_layout: cute.ComposedLayout
        :param epi_smem_layout: Shared memory layout for epilogue
        :type epi_smem_layout: cute.ComposedLayout
        """

        varlen_m = const_expr(varlen_params.cu_seqlens_m is not None)
        varlen_k = const_expr(varlen_params.cu_seqlens_k is not None)
        assert not (varlen_m and varlen_k)
        if const_expr(self.gather_A):
            assert varlen_m or varlen_k
        has_D = const_expr(mD_mnl is not None)
        has_C = const_expr(mC_mnl is not None)

        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())

        # /////////////////////////////////////////////////////////////////////////////
        #  Prefetch Tma desc
        # /////////////////////////////////////////////////////////////////////////////
        if warp_idx == self.ab_load_warp_id:
            for tma_atom in (tma_atom_a, tma_atom_b, tma_atom_d, tma_atom_c):
                if const_expr(tma_atom is not None):
                    cpasync.prefetch_descriptor(tma_atom)

        # /////////////////////////////////////////////////////////////////////////////
        #  Alloc and init AB full/empty + ACC full mbar (pipeline)
        # /////////////////////////////////////////////////////////////////////////////
        smem = cutlass.utils.SmemAllocator()
        storage = smem.allocate(self.shared_storage)

        ab_pipeline = self.make_ab_pipeline(
            tiled_mma=tiled_mma,
            cluster_layout_vmnk=cute.make_layout((1, *cluster_layout_mnk.shape)),
            ab_pipeline_mbar_ptr=storage.ab_pipeline_array_ptr.data_ptr(),
        )
        epi_pipeline = None
        if const_expr(has_C):
            epi_pipeline = self.make_epi_pipeline(
                c_smem_layout=cute.slice_(epi_c_smem_layout, (None, None, 0)),
                epi_pipeline_mbar_ptr=storage.epi_pipeline_array_ptr.data_ptr(),
            )
        sched_pipeline = None
        tile_count = None
        if const_expr(tile_sched_params.tile_count_semaphore is not None):
            # Dynamic persistent scheduler
            sched_pipeline = self.make_sched_pipeline(
                cluster_layout_mnk,
                sched_pipeline_mbar_ptr=storage.sched_pipeline_array_ptr.data_ptr(),
                varlen_k=varlen_k,
            )
            tile_count = storage.tile_count.get_tensor((self.sched_stage,))

        # ///////////////////////////////////////////////////////////////////////////////
        #  Generate smem tensor A/B
        # ///////////////////////////////////////////////////////////////////////////////
        sA = storage.sA.get_tensor(a_smem_layout.outer, swizzle=a_smem_layout.inner)
        sB = storage.sB.get_tensor(b_smem_layout.outer, swizzle=b_smem_layout.inner)
        sD = None
        if const_expr(has_D):
            if const_expr(not self.is_persistent):
                sD_ptr = cute.recast_ptr(sA.iterator, epi_smem_layout.inner, dtype=self.d_dtype)
                sD = cute.make_tensor(sD_ptr, epi_smem_layout.outer)
            else:
                sD = storage.sD.get_tensor(epi_smem_layout.outer, swizzle=epi_smem_layout.inner)
        sC = None
        if const_expr(has_C):
            sC = storage.sC.get_tensor(epi_c_smem_layout.outer, swizzle=epi_c_smem_layout.inner)
        epi_smem_tensors = self.epi_get_smem_tensors(epilogue_params, storage)

        varlen_manager = VarlenManager.create(
            varlen_params,
            has_D,
            self.num_epi_tensormaps,
            # Only used if not varlen_m
            len_m_static=Int32(
                mA_mkl.shape[0]
                if varlen_k or varlen_params.mAIdx is None
                else varlen_params.mAIdx.shape[0]
            ),
            len_k_static=Int32(mA_mkl.shape[1]),
            pingpong=self.pingpong,
            warp_idx=warp_idx,
        )

        TileSchedulerCls = partial(
            TileSchedulerCls.create, tile_sched_params, tile_count, sched_pipeline
        )

        if warp_idx >= self.ab_load_warp_id:
            cute.arch.warpgroup_reg_dealloc(self.num_regs_load)
            if (
                warp_idx >= self.ab_load_warp_id
                and warp_idx < self.ab_load_warp_id + self.num_ab_load_warps
            ):
                is_tma_warp = self.num_ab_load_warps == 1 or warp_idx == self.ab_load_warp_id
                # initialize tensormap for A & B
                varlen_manager.init_tensormap_AB(tma_atom_a, tma_atom_b, is_tma_warp)
                tma_desc_a_ptr = varlen_manager.get_tma_desc_a_ptr()
                tma_desc_b_ptr = varlen_manager.get_tma_desc_b_ptr()
                # ///////////////////////////////////////////////////////////////////////////////
                # Get mcast mask
                # ///////////////////////////////////////////////////////////////////////////////
                cta_rank_in_cluster = cute.arch.make_warp_uniform(cute.arch.block_idx_in_cluster())
                block_in_cluster_coord_mnk = cluster_layout_mnk.get_flat_coord(cta_rank_in_cluster)
                a_mcast_mask = cute.make_layout_image_mask(
                    cluster_layout_mnk, block_in_cluster_coord_mnk, mode=1
                )
                b_mcast_mask = cute.make_layout_image_mask(
                    cluster_layout_mnk, block_in_cluster_coord_mnk, mode=0
                )
                a_mcast_mask = a_mcast_mask if self.is_a_mcast else 0
                b_mcast_mask = b_mcast_mask if self.is_b_mcast else 0

                # Persistent tile scheduling loop
                is_scheduler_warp = self.num_ab_load_warps == 1 or warp_idx == self.ab_load_warp_id
                if const_expr(cute.size(cluster_layout_mnk) > 1):
                    is_scheduler_warp = is_scheduler_warp and cute.arch.block_idx_in_cluster() == 0
                tile_scheduler = TileSchedulerCls(is_scheduler_warp=is_scheduler_warp)
                work_tile = tile_scheduler.initial_work_tile_info()
                ab_producer_state = make_pipeline_state(
                    pipeline.PipelineUserType.Producer, self.ab_stage
                )
                if const_expr(varlen_k):
                    # wait tensormap initialization complete before update
                    varlen_manager.fence_tensormap_init()
                while work_tile.is_valid_tile:
                    tile_coord_mnkl = work_tile.tile_idx
                    batch_idx = tile_coord_mnkl[3]
                    varlen_manager.update_tensormap_AB(
                        batch_idx,
                        self.a_layout,
                        self.b_layout,
                        is_tma_warp,
                    )
                    # ///////////////////////////////////////////////////////////////////////////
                    #  Local_tile partition global tensors
                    # ///////////////////////////////////////////////////////////////////////////
                    if const_expr(not self.gather_A):
                        mA_mk = varlen_manager.offset_batch_A(mA_mkl, batch_idx)
                        # (bM, bK, RestK)
                        gA_mk = cute.local_tile(
                            mA_mk,
                            cute.select(self.cta_tile_shape_mnk, [0, 2]),
                            (tile_coord_mnkl[0], None),
                        )
                    else:
                        mAIdx_mk = varlen_manager.offset_batch_AIdx(batch_idx)
                        if const_expr(varlen_m):
                            gAIdx = cute.local_tile(
                                mAIdx_mk, (self.cta_tile_shape_mnk[0],), (tile_coord_mnkl[0],)
                            )
                            # (M, K)
                            mA_mk = mA_mkl
                        else:
                            assert varlen_k
                            # (tile_K, RestK)
                            gAIdx = cute.flat_divide(mAIdx_mk, (self.cta_tile_shape_mnk[2],))
                            # (tile_M, K)
                            mA_mk = cute.local_tile(
                                mA_mkl, (self.cta_tile_shape_mnk[0],), (tile_coord_mnkl[0], None)
                            )
                    # (bN, bK, RestK)
                    gB_nk = cute.local_tile(
                        varlen_manager.offset_batch_B(mB_nkl, batch_idx),
                        cute.select(self.cta_tile_shape_mnk, [1, 2]),
                        (tile_coord_mnkl[1], None),
                    )
                    # //////////////////////////////////////////////////////////////////////////
                    #  Partition shared tensor for TMA load A/B
                    # //////////////////////////////////////////////////////////////////////////
                    varlen_manager.fence_tensormap_update_AB(is_tma_warp)
                    len_m = varlen_manager.len_m(batch_idx)
                    len_k = varlen_manager.len_k(batch_idx)
                    #  TMA load A partition_S/D
                    copy_A = None
                    if const_expr(not self.gather_A):
                        copy_A, _, _ = copy_utils_tma_get_copy_fn(
                            tma_atom_a,
                            cta_coord=block_in_cluster_coord_mnk[1],
                            cta_layout=cute.make_layout(
                                cute.slice_(cluster_layout_mnk, (0, None, 0)).shape
                            ),
                            src_tensor=gA_mk,
                            dst_tensor=sA,
                            mcast_mask=a_mcast_mask,
                            tma_desc_ptr=tma_desc_a_ptr,
                        )
                    else:
                        tiled_copy_A = self._make_gmem_tiled_copy_A(
                            mA_mkl.element_type, self.a_layout, self.num_ab_load_warps * 32
                        )
                        tidx = (
                            cute.arch.thread_idx()[0] - cute.arch.WARP_SIZE * self.ab_load_warp_id
                        )
                        thr_copy_A = tiled_copy_A.get_slice(tidx)
                        copy_A, prefetch_A = None, None
                        if const_expr(varlen_m):
                            copy_A = copy_utils_gather_m_get_copy_fn(
                                thr_copy_A,
                                mA_mk,
                                sA,
                                gAIdx,
                                limit_m=len_m - tile_coord_mnkl[0] * self.cta_tile_shape_mnk[0],
                                limit_k=len_k,
                            )
                        else:
                            copy_A, prefetch_A = copy_utils_gather_k_get_copy_fn(
                                thr_copy_A,
                                mA_mk,
                                sA,
                                gAIdx,
                                limit_m=len_m - tile_coord_mnkl[0] * self.cta_tile_shape_mnk[0],
                                limit_k=len_k,
                            )
                    # TMA load B partition_S/D
                    copy_B, _, _ = copy_utils_tma_get_copy_fn(
                        tma_atom_b,
                        cta_coord=block_in_cluster_coord_mnk[0],
                        cta_layout=cute.make_layout(
                            cute.slice_(cluster_layout_mnk, (None, 0, 0)).shape
                        ),
                        src_tensor=gB_nk,
                        dst_tensor=sB,
                        mcast_mask=b_mcast_mask,
                        tma_desc_ptr=tma_desc_b_ptr,
                    )
                    k_tile_cnt = cute.ceil_div(len_k, self.cta_tile_shape_mnk[2])
                    if const_expr(not self.gather_A):
                        ab_producer_state = self.load_AB(
                            ab_pipeline, ab_producer_state, copy_A, copy_B, k_tile_cnt
                        )
                    else:
                        ab_producer_state = self.load_AB_gather_A(
                            ab_pipeline,
                            ab_producer_state,
                            copy_A,
                            prefetch_A,
                            copy_B,
                            k_tile_cnt,
                            varlen_m=varlen_m,
                        )
                    tile_scheduler.fetch_next_work(is_scheduler_warp=is_scheduler_warp)
                    tile_scheduler.advance_to_next_work(is_scheduler_warp=is_scheduler_warp)
                    work_tile = tile_scheduler.get_current_work()
                    # End of persistent scheduler loop
                if const_expr(self.pingpong and not varlen_k):
                    # Need to write the tile_idx to smem for the next WG in the pingpong mode
                    tile_scheduler.advance_to_next_work(is_scheduler_warp=is_scheduler_warp)
                ab_pipeline.producer_tail(ab_producer_state)
                if is_scheduler_warp:
                    tile_scheduler.producer_tail()

        if warp_idx < self.ab_load_warp_id:
            cute.arch.warpgroup_reg_alloc(self.num_regs_mma)
            is_tma_warp = Boolean(
                (not self.pingpong and warp_idx == 0)
                or (self.pingpong and (warp_idx == 0 or warp_idx == 4))
            )
            varlen_manager.init_tensormap_epi(
                tma_atom_d, self.epi_get_tma_atoms(epilogue_params), is_tma_warp
            )
            tma_desc_d_ptr = varlen_manager.get_tma_desc_d_ptr()
            tma_desc_epi_ptrs = varlen_manager.get_tma_desc_epi_ptrs()
            # //////////////////////////////////////////////////////////////////////////////
            #  Partition global tensor for TiledMMA_A/B/C
            # //////////////////////////////////////////////////////////////////////////////
            tidx, _, _ = cute.arch.thread_idx()
            warp_group_idx = cute.arch.make_warp_uniform(tidx // self.num_threads_per_warp_group)
            if const_expr(self.pingpong):
                tidx = tidx % self.num_threads_per_warp_group
            warp_group_thread_layout = cute.make_layout(
                self.mma_warp_groups if not self.pingpong else 1,
                stride=self.num_threads_per_warp_group,
            )
            thr_mma = tiled_mma.get_slice(
                warp_group_thread_layout(warp_group_idx if not self.pingpong else 0)
            )

            # //////////////////////////////////////////////////////////////////////////////
            #  Make fragments
            # //////////////////////////////////////////////////////////////////////////////
            tCrA = tiled_mma.make_fragment_A(thr_mma.partition_A(sA))
            tCrB = tiled_mma.make_fragment_B(thr_mma.partition_B(sB))

            acc_shape = tiled_mma.partition_shape_C(
                cute.select(self.cta_tile_shape_mnk, mode=[0, 1])
            )
            acc = cute.make_fragment(acc_shape, self.acc_dtype)
            acc_slow = None
            if const_expr(self.fp8_slow_accum):
                acc_slow = cute.make_fragment(acc_shape, self.acc_dtype)

            if const_expr(self.pingpong):
                if warp_group_idx == 0:
                    # WG0 needs a start signal at the very beginning
                    self.pingpong_barrier_arrive(warp_group_idx=0, stage="mma")
                    self.pingpong_barrier_arrive(warp_group_idx=0, stage="epi")

            k_tile_cnt_static = cute.ceil_div(mA_mkl.shape[1], self.cta_tile_shape_mnk[2])
            c_tile_cnt = cute.size(cute.ceil_div(self.cta_tile_shape_mnk[:2], self.epi_tile))

            ab_read_state = make_pipeline_state(pipeline.PipelineUserType.Consumer, self.ab_stage)
            epi_store_pipeline = self.make_epi_store_pipeline()
            epi_read_state = make_pipeline_state(
                pipeline.PipelineUserType.Consumer, self.epi_c_stage
            )
            epi_producer_state = make_pipeline_state(
                pipeline.PipelineUserType.Producer, self.epi_c_stage
            )
            tile_scheduler = TileSchedulerCls()
            work_tile = None
            if const_expr(self.pingpong):
                if const_expr(varlen_k):
                    work_tile = tile_scheduler.initial_work_tile_info()
                if warp_idx >= 4:
                    # Advance 2nd Math WG pipeline states to the end of 1st Math WG
                    epi_read_state.advance_iters(c_tile_cnt)
                    epi_producer_state.advance_iters(c_tile_cnt)
                    if const_expr(not varlen_k):
                        ab_read_state.advance_iters(k_tile_cnt_static)
                    else:
                        len_k = varlen_manager.len_k(batch_idx=work_tile.tile_idx[3])
                        k_tile_cnt = cute.ceil_div(len_k, self.cta_tile_shape_mnk[2])
                        ab_read_state.advance_iters(k_tile_cnt)
                    tile_scheduler.advance_to_next_work()
                    if const_expr(varlen_k):
                        work_tile = tile_scheduler.get_current_work()
                if const_expr(not varlen_k):
                    work_tile = tile_scheduler.initial_work_tile_info()
            else:
                work_tile = tile_scheduler.initial_work_tile_info()
            if const_expr(varlen_m):
                # wait tensormap initialization complete before update
                varlen_manager.fence_tensormap_init()
            while work_tile.is_valid_tile:
                tile_coord_mnkl = work_tile.tile_idx
                batch_idx = tile_coord_mnkl[3]
                epi_shapes, epi_orders = self.epi_get_tensormap_update_shapes_orders(
                    epilogue_params, varlen_params.cu_seqlens_m, batch_idx
                )
                varlen_manager.update_tensormap_epi(
                    batch_idx,
                    self.d_layout,
                    epi_shapes,
                    epi_orders,
                    is_tma_warp,
                )
                len_k = varlen_manager.len_k(batch_idx)
                k_tile_cnt = cute.ceil_div(len_k, self.cta_tile_shape_mnk[2])
                ab_read_state, tiled_mma = self.mma(
                    ab_pipeline,
                    ab_read_state,
                    tiled_mma,
                    tCrA,
                    tCrB,
                    acc,
                    acc_slow,
                    k_tile_cnt,
                    warp_group_idx,
                )
                if const_expr(varlen_k):
                    if k_tile_cnt == 0:
                        acc.fill(0.0)

                # /////////////////////////////////////////////////////////////////////////////
                #  EPILOGUE
                # /////////////////////////////////////////////////////////////////////////////
                if const_expr(self.pingpong):
                    self.pingpong_barrier_sync(warp_group_idx, "epi")

                epilogue_barrier = pipeline.NamedBarrier(
                    barrier_id=int(NamedBarrierGemm.Epilogue),
                    num_threads=self.num_epi_warps * cute.arch.WARP_SIZE,
                )

                varlen_manager.fence_tensormap_update_epi(is_tma_warp)

                copy_D = None
                if const_expr(has_D):
                    copy_D, _, _ = self.epilog_gmem_copy_and_partition(
                        tma_atom_d,
                        varlen_manager.offset_batch_epi(mD_mnl, batch_idx),
                        self.cta_tile_shape_mnk[:2],
                        self.epi_tile,
                        sD,
                        tile_coord_mnkl,
                        tma_desc_ptr=tma_desc_d_ptr,
                    )
                copy_C = None
                if const_expr(has_C):
                    copy_C_fn, _, _ = self.epilog_gmem_copy_and_partition(
                        tma_atom_c,
                        varlen_manager.offset_batch_epi(mC_mnl, batch_idx),
                        self.cta_tile_shape_mnk[:2],
                        self.epi_tile,
                        sC,
                        tile_coord_mnkl,
                    )
                    copy_C = copy_utils.tma_producer_copy_fn(copy_C_fn, epi_pipeline)

                d_dtype_for_layout = self.d_dtype if self.d_dtype is not None else cutlass.BFloat16
                tiled_copy_r2s, tRS_rD, tRS_sD = self.epilog_smem_store_and_partition(
                    tiled_mma, self.d_layout, d_dtype_for_layout, sD, tidx
                )
                # (R2S, R2S_M, R2S_N)
                tRS_rAcc = tiled_copy_r2s.retile(acc)
                load_acc_subtile = partial(self.epi_load_acc_subtile, tRS_rAcc)
                if const_expr(has_C):
                    tiled_copy_s2r, tRS_rC, tSR_rC, tSR_sC = self.epilog_smem_load_and_partition(
                        tiled_mma, self.c_layout, self.c_dtype, sC, tRS_rD.layout, tidx
                    )
                else:
                    tiled_copy_s2r, tSR_sC, tRS_rC, tSR_rC = None, None, None, None

                # Wait for all warp groups in the thread block to finish, because smem for tensor
                # A in the mainloop is reused in the epilogue if not persistent.
                if const_expr(not self.is_persistent):
                    epilogue_barrier.arrive_and_wait()

                self.epi_visit_acc(epilogue_params, acc, tiled_mma, tile_coord_mnkl, tidx)

                epi_read_state, epi_producer_state = self.epilogue(
                    epilogue_params,
                    epi_smem_tensors,
                    tma_desc_epi_ptrs,
                    epi_pipeline,
                    epi_store_pipeline,
                    epi_read_state,
                    epi_producer_state,
                    self.epi_tile,
                    load_acc_subtile,
                    tRS_rD,
                    tRS_rC,
                    None,  # tiled_copy_t2r, for Sm100 only
                    tiled_copy_r2s,
                    tRS_sD,
                    tiled_copy_s2r,
                    tSR_rC,
                    tSR_sC,
                    copy_D,
                    copy_C,
                    tile_coord_mnkl,
                    varlen_manager,
                    epilogue_barrier,
                    tile_scheduler,
                    tidx,
                    is_tma_warp,
                )

                if const_expr(self.pingpong):
                    # With pingpong, 2 WGs write two different output tiles to the same smem,
                    # so we have to make sure the smem content is done reading before signaling
                    # the next WG's epilogue.
                    if is_tma_warp:
                        epi_store_pipeline.producer_tail()
                    self.pingpong_barrier_arrive(1 - warp_group_idx, stage="epi")

                if const_expr(not self.pingpong):
                    tile_scheduler.advance_to_next_work()
                    work_tile = tile_scheduler.get_current_work()
                else:  # Skip a tile for pingpong
                    # Update starting load/store pipeline states for the next tile
                    epi_read_state.advance_iters(c_tile_cnt)
                    epi_producer_state.advance_iters(c_tile_cnt)
                    # Update starting mainloop pipeline state for the next tile
                    if const_expr(not varlen_k):
                        ab_read_state.advance_iters(k_tile_cnt_static)
                        tile_scheduler.advance_to_next_work(advance_count=self.mma_warp_groups)
                        work_tile = tile_scheduler.get_current_work()
                    else:
                        tile_scheduler.advance_to_next_work()
                        work_tile = tile_scheduler.get_current_work()
                        if work_tile.is_valid_tile:
                            len_k = varlen_manager.len_k(batch_idx=work_tile.tile_idx[3])
                            k_tile_cnt = cute.ceil_div(len_k, self.cta_tile_shape_mnk[2])
                            ab_read_state.advance_iters(k_tile_cnt)
                            tile_scheduler.advance_to_next_work()
                            work_tile = tile_scheduler.get_current_work()
                # End of persistent scheduler loop

            # Wait for D store complete
            if const_expr(not self.pingpong):
                if is_tma_warp:
                    epi_store_pipeline.producer_tail()

    @cute.jit
    def load_AB(
        self,
        ab_pipeline: cutlass.pipeline.PipelineAsync,
        ab_producer_state: cutlass.pipeline.PipelineState,
        copy_A: Optional[Callable],
        copy_B: Callable,
        k_tile_cnt: Int32,
        # These are for Sm100 blockscaled gemm
        copy_SFA: Optional[Callable] = None,
        copy_SFB: Optional[Callable] = None,
    ) -> cutlass.pipeline.PipelineState:
        blockscaled = const_expr(copy_SFA is not None)
        if const_expr(blockscaled):
            assert copy_SFB is not None
        # Peek (try_wait) AB buffer empty for k_block = prefetch_k_tile_cnt
        peek_ab_empty_status = Boolean(True)
        if 0 < k_tile_cnt:
            peek_ab_empty_status = ab_pipeline.producer_try_acquire(ab_producer_state)
        # /////////////////////////////////////////////////////////////////////////
        # TMA load
        # /////////////////////////////////////////////////////////////////////////
        for k_tile in cutlass.range(k_tile_cnt, unroll=1):
            # Wait for A/B buffers to be empty before loading into them
            # Also sets the transaction barrier for the A/B buffers
            ab_pipeline.producer_acquire(ab_producer_state, peek_ab_empty_status)
            tma_bar_ptr = ab_pipeline.producer_get_barrier(ab_producer_state)
            smem_idx = ab_producer_state.index
            if const_expr(copy_A is not None):
                copy_A(k_tile, smem_idx, tma_bar_ptr=tma_bar_ptr)
            copy_B(k_tile, smem_idx, tma_bar_ptr=tma_bar_ptr)
            if const_expr(blockscaled):
                copy_SFA(k_tile, smem_idx, tma_bar_ptr=tma_bar_ptr)
                copy_SFB(k_tile, smem_idx, tma_bar_ptr=tma_bar_ptr)
            # Mainloop pipeline's producer commit is a NOP
            ab_pipeline.producer_commit(ab_producer_state)
            ab_producer_state.advance()
            peek_ab_empty_status = Boolean(True)
            if k_tile + 1 < k_tile_cnt:
                peek_ab_empty_status = ab_pipeline.producer_try_acquire(ab_producer_state)
        return ab_producer_state

    @cute.jit
    def load_AB_gather_A(
        self,
        ab_pipeline: cutlass.pipeline.PipelineAsync,
        ab_producer_state: cutlass.pipeline.PipelineState,
        copy_A: Callable,
        prefetch_A: Optional[Callable],
        copy_B: Callable,
        k_tile_cnt: Int32,
        varlen_m: bool = True,
    ) -> cutlass.pipeline.PipelineState:
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        # Peek (try_wait) AB buffer empty for k_block = prefetch_k_tile_cnt
        peek_ab_empty_status = Boolean(True)
        if 0 < k_tile_cnt:
            peek_ab_empty_status = ab_pipeline.producer_try_acquire(ab_producer_state)
        # /////////////////////////////////////////////////////////////////////////
        # TMA load on B and cp.async on A
        # /////////////////////////////////////////////////////////////////////////
        for k_tile in cutlass.range(k_tile_cnt - 1, unroll=1):
            prefetch_out = ()
            if const_expr(prefetch_A is not None):  # Prefetch early, even before smem is free
                prefetch_out = (prefetch_A(k_tile),)
            # Wait for A/B buffers to be empty before loading into them
            # Also sets the transaction barrier for the A/B buffers
            # A tiny bit faster to rotate the warp that does TMA
            # However, for varlen_k, we must use the warp_idx == self.ab_load_warp_id
            # since that's the warp that does the tensormap update.
            is_tma_warp = warp_idx == self.ab_load_warp_id + (
                (k_tile % self.num_ab_load_warps) if const_expr(varlen_m) else 0
            )
            ab_pipeline.producer_acquire(ab_producer_state, peek_ab_empty_status, is_tma_warp)
            smem_idx = ab_producer_state.index
            # A bit faster to load B first while we calculate the indices for A
            if is_tma_warp:
                tma_bar_ptr = ab_pipeline.producer_get_barrier(ab_producer_state)
                copy_B(k_tile, smem_idx, tma_bar_ptr=tma_bar_ptr)
            copy_A(k_tile, smem_idx, *prefetch_out)
            # This tells mbarrier to track the completion of cp.async
            ab_pipeline.producer_cpasync_commit(ab_producer_state)
            ab_producer_state.advance()
            peek_ab_empty_status = Boolean(True)
            if k_tile + 1 < k_tile_cnt:
                peek_ab_empty_status = ab_pipeline.producer_try_acquire(ab_producer_state)
        # bound checking in the K dimension on the last k_tile
        if 0 < k_tile_cnt:
            k_tile = k_tile_cnt - 1
            prefetch_out = ()
            if const_expr(prefetch_A is not None):  # Prefetch early, even before smem is free
                prefetch_out = (prefetch_A(k_tile, pred=True),)
            is_tma_warp = warp_idx == self.ab_load_warp_id + (
                (k_tile % self.num_ab_load_warps) if const_expr(varlen_m) else 0
            )
            ab_pipeline.producer_acquire(ab_producer_state, peek_ab_empty_status, is_tma_warp)
            smem_idx = ab_producer_state.index
            if is_tma_warp:
                tma_bar_ptr = ab_pipeline.producer_get_barrier(ab_producer_state)
                copy_B(k_tile, smem_idx, tma_bar_ptr=tma_bar_ptr)
            copy_A(k_tile, smem_idx, *prefetch_out, pred=True)
            ab_pipeline.producer_cpasync_commit(ab_producer_state)
            ab_producer_state.advance()
        return ab_producer_state

    @cute.jit
    def mma(
        self,
        ab_pipeline: cutlass.pipeline.PipelineAsync,
        ab_read_state: cutlass.pipeline.PipelineState,
        tiled_mma: cute.TiledMma,
        tCrA: cute.Tensor,
        tCrB: cute.Tensor,
        acc: cute.Tensor,
        acc_slow: Optional[cute.Tensor],
        k_tile_cnt: Int32,
        warp_group_idx: Int32,
    ) -> Tuple[cutlass.pipeline.PipelineState, cute.TiledMma]:
        # /////////////////////////////////////////////////////////////////////////////
        #  Prologue MMAs
        # /////////////////////////////////////////////////////////////////////////////
        k_pipe_mmas = 1
        ab_release_state = ab_read_state.clone()
        num_prologue_mma = min(k_pipe_mmas, k_tile_cnt)
        if const_expr(self.pingpong):
            self.pingpong_barrier_sync(warp_group_idx, stage="mma")
        peek_ab_full_status = Boolean(True)
        if 0 < k_tile_cnt:
            peek_ab_full_status = ab_pipeline.consumer_try_wait(ab_read_state)
        tiled_mma.set(warpgroup.Field.ACCUMULATE, False)
        num_k_blocks = cute.size(tCrA, mode=[2])
        for k_tile in cutlass.range(num_prologue_mma):
            # Wait for A/B buffer to be ready
            ab_pipeline.consumer_wait(ab_read_state, peek_ab_full_status)
            warpgroup.fence()
            for k_blk_idx in cutlass.range(num_k_blocks, unroll_full=True):
                k_blk_coord = (None, None, k_blk_idx, ab_read_state.index)
                cute.gemm(tiled_mma, acc, tCrA[k_blk_coord], tCrB[k_blk_coord], acc)
                tiled_mma.set(warpgroup.Field.ACCUMULATE, True)
            warpgroup.commit_group()
            ab_read_state.advance()
            peek_ab_full_status = Boolean(True)
            if k_tile + 1 < k_tile_cnt:
                peek_ab_full_status = ab_pipeline.consumer_try_wait(ab_read_state)
        # If k_tile_cnt == 0, this is not correct. But we will set acc to 0 in the mainloop
        # in that case.
        if const_expr(self.fp8_slow_accum):
            warpgroup.wait_group(0)
            acc_slow.store(acc.load())

        # /////////////////////////////////////////////////////////////////////////////
        #  MAINLOOP
        # /////////////////////////////////////////////////////////////////////////////
        for k_tile in cutlass.range(num_prologue_mma, k_tile_cnt, unroll=1):
            # Wait for TMA copies to complete
            ab_pipeline.consumer_wait(ab_read_state, peek_ab_full_status)
            # WGMMA
            warpgroup.fence()
            if const_expr(self.fp8_slow_accum):
                tiled_mma.set(warpgroup.Field.ACCUMULATE, False)
            for k_blk_idx in cutlass.range(num_k_blocks, unroll_full=True):
                k_blk_coord = (None, None, k_blk_idx, ab_read_state.index)
                cute.gemm(tiled_mma, acc, tCrA[k_blk_coord], tCrB[k_blk_coord], acc)
                tiled_mma.set(warpgroup.Field.ACCUMULATE, True)
            warpgroup.commit_group()
            # Wait on the wgmma barrier for previous k_pipe_mmas wgmmas to complete
            if const_expr(not self.fp8_slow_accum):
                warpgroup.wait_group(k_pipe_mmas)
            else:
                warpgroup.wait_group(0)
                acc_slow.store(acc_slow.load() + acc.load())
            ab_pipeline.consumer_release(ab_release_state)
            ab_read_state.advance()
            ab_release_state.advance()
            peek_ab_full_status = Boolean(True)
            if k_tile + 1 < k_tile_cnt:
                peek_ab_full_status = ab_pipeline.consumer_try_wait(ab_read_state)
        if const_expr(self.pingpong):
            # Cue for next WG's MMA to start
            self.pingpong_barrier_arrive(1 - warp_group_idx, stage="mma")
        if const_expr(not self.fp8_slow_accum):
            # fp8_slow_accum would already called wait_group(0) inside the loop
            warpgroup.wait_group(0)
        for k_tile in cutlass.range(num_prologue_mma, unroll=1):
            ab_pipeline.consumer_release(ab_release_state)
            ab_release_state.advance()
        if const_expr(self.fp8_slow_accum):
            acc.store(acc_slow.load())
        # If we don't return the tiled_mma, we get compiler error
        # "operand #0 does not dominate this use"
        return ab_read_state, tiled_mma

    @cute.jit
    def epilogue(
        self,
        params: EpilogueParams,
        epi_smem_tensors: Tuple[cute.Tensor, ...],
        tma_desc_epi_ptrs: list[Optional[cute.Pointer]],
        epi_pipeline: cutlass.pipeline.PipelineAsync,
        epi_store_pipeline: cutlass.pipeline.PipelineAsync,
        epi_read_state: cutlass.pipeline.PipelineState,
        epi_producer_state: Optional[cutlass.pipeline.PipelineState],
        epi_tile: cute.Tile,
        load_acc_subtile: Callable,
        tRS_rD: cute.Tensor,
        tRS_rC: Optional[cute.Tensor],
        tiled_copy_t2r: Optional[cute.TiledCopy],  # Only for Sm100
        tiled_copy_r2s: cute.TiledCopy,
        tRS_sD: cute.Tensor,
        tiled_copy_s2r: Optional[cute.ThrCopy],
        tSR_rC: Optional[cute.Tensor],
        tSR_sC: Optional[cute.Tensor],
        copy_D: Optional[Callable],
        copy_C: Optional[Callable],
        tile_coord_mnkl: cute.Coord,
        varlen_manager: VarlenManager,
        epilogue_barrier: cutlass.pipeline.NamedBarrier,
        tile_scheduler,
        tidx: Int32,
        is_tma_warp: Boolean,
    ) -> Tuple[cutlass.pipeline.PipelineState, cutlass.pipeline.PipelineState]:
        has_C = const_expr(tRS_rC is not None)
        has_D = const_expr(copy_D is not None)
        epi_tile_shape = cute.zipped_divide(
            cute.make_layout(self.cta_tile_shape_mnk[:2]), epi_tile
        ).shape[1]
        # We iterate over epi tiles in the N dimension first before the M dimension
        epi_tile_layout = cute.make_ordered_layout(epi_tile_shape, order=(1, 0))
        epi_tile_num = cute.size(epi_tile_shape)
        num_prev_subtiles = tile_scheduler.num_tiles_executed * epi_tile_num

        epi_tensors = self.epi_begin(
            params,
            epi_smem_tensors,
            epi_tile,
            tiled_copy_t2r,
            tiled_copy_r2s,
            tile_coord_mnkl,
            varlen_manager,
            epilogue_barrier,
            tidx,
        )

        if const_expr(copy_C is not None):
            for epi_idx in cutlass.range(min(epi_tile_num, self.epi_c_stage), unroll=1):
                gmem_coord_C = epi_tile_layout.get_hier_coord(epi_idx)
                if is_tma_warp:
                    epi_pipeline.producer_acquire(epi_producer_state)
                    copy_C(src_idx=gmem_coord_C, producer_state=epi_producer_state)
                    epi_pipeline.producer_commit(epi_producer_state)
                epi_producer_state.advance()

        def tma_store_fn(src_idx, dst_idx):
            # Fence and barrier to make sure shared memory store is visible to TMA store
            cute.arch.fence_proxy(
                cute.arch.ProxyKind.async_shared, space=cute.arch.SharedSpace.shared_cta
            )
            epilogue_barrier.arrive_and_wait()
            # Copy from shared memory to global memory
            if is_tma_warp:
                if const_expr(has_D):
                    copy_D(src_idx=src_idx, dst_idx=dst_idx)
            # Can't use if statement here, epi_store_pipeline object isn't captured somehow
            if_generate(is_tma_warp, lambda: epi_store_pipeline.producer_commit())
            if_generate(is_tma_warp, lambda: epi_store_pipeline.producer_acquire())
            epilogue_barrier.arrive_and_wait()

        # We could delay the TMA store by 1 epi tile to better overlap the non-TMA ops
        # with the TMA store. However, currently this doesn't seem to improve perf.
        delay_tma_store = False

        src_idx_prev, dst_idx_prev = None, None
        for epi_idx in cutlass.range_constexpr(epi_tile_num):
            # The global memory coordinate for the current epi tile
            gmem_coord = epi_tile_layout.get_hier_coord(epi_idx)
            # Copy from acc to D registers
            load_acc_subtile(tRS_rD, epi_idx)
            epi_loop_tensors = self.epi_begin_loop(params, epi_tensors, gmem_coord)
            if const_expr(has_C):
                epi_pipeline.consumer_wait(epi_read_state)
                cute.copy(tiled_copy_s2r, tSR_sC[None, None, None, epi_read_state.index], tSR_rC)
                # Fence to make sure shared memory read is visible to TMA load
                cute.arch.fence_proxy(
                    cute.arch.ProxyKind.async_shared, space=cute.arch.SharedSpace.shared_cta
                )
                cute.arch.sync_warp()
                with cute.arch.elect_one():
                    epi_pipeline.consumer_release(epi_read_state)
                epi_read_state.advance()
            if const_expr(copy_C is not None and epi_idx + self.epi_c_stage < epi_tile_num):
                gmem_coord_C = epi_tile_layout.get_hier_coord(epi_idx + self.epi_c_stage)
                if is_tma_warp:
                    epi_pipeline.producer_acquire(epi_producer_state)
                    copy_C(src_idx=gmem_coord_C, producer_state=epi_producer_state)
                    epi_pipeline.producer_commit(epi_producer_state)
                epi_producer_state.advance()
            tRS_rEpi = self.epi_visit_subtile(params, epi_loop_tensors, tRS_rD, tRS_rC)
            epi_buffer = (num_prev_subtiles + epi_idx) % self.epi_stage
            if const_expr(delay_tma_store):
                if const_expr(epi_idx > 0):
                    tma_store_fn(src_idx=src_idx_prev, dst_idx=dst_idx_prev)
                src_idx_prev, dst_idx_prev = epi_buffer, gmem_coord
            # Copy from D registers to shared memory
            if const_expr(has_D):
                copy_utils_cvt_copy(tiled_copy_r2s, tRS_rD, tRS_sD[None, None, None, epi_buffer])
            if const_expr(not delay_tma_store):
                tma_store_fn(src_idx=epi_buffer, dst_idx=gmem_coord)

        if const_expr(delay_tma_store):
            tma_store_fn(src_idx=src_idx_prev, dst_idx=dst_idx_prev)

        self.epi_end(
            params,
            epi_tensors,
            epi_tile,
            tiled_copy_t2r,
            tiled_copy_r2s,
            tile_coord_mnkl,
            varlen_manager,
            tidx,
        )

        return epi_read_state, epi_producer_state

    def get_scheduler_class(self, varlen_m: bool = False):
        """Return the scheduler class to use. Override in subclasses for custom schedulers."""
        return TileScheduler if not varlen_m else VarlenMTileScheduler

    def get_scheduler_arguments(
        self,
        mA: cute.Tensor,
        mB: cute.Tensor,
        mD: Optional[cute.Tensor],
        scheduler_args,
        varlen_args,
    ):
        """Create scheduler arguments. Override in subclasses for custom schedulers."""
        if const_expr(varlen_args.mCuSeqlensM is None):
            num_problems = (
                mD.shape[2]
                if mD is not None
                else (
                    mB.shape[2]
                    if varlen_args.mCuSeqlensK is None
                    else varlen_args.mCuSeqlensK.shape[0] - 1
                )
            )
            problem_shape_ntile_mnl = (
                cute.ceil_div(mA.shape[0], self.cta_tile_shape_mnk[0]),
                cute.ceil_div(mB.shape[0], self.cta_tile_shape_mnk[1]),
                num_problems,
            )
            tile_sched_args = TileSchedulerArguments(
                problem_shape_ntile_mnl=problem_shape_ntile_mnl,
                raster_order=scheduler_args.raster_order,
                group_size=scheduler_args.max_swizzle_size,
                cluster_shape_mnk=self.cluster_shape_mnk,
                tile_count_semaphore=scheduler_args.tile_count_semaphore,
                batch_idx_permute=scheduler_args.batch_idx_permute,
                is_persistent=self.is_persistent,
            )
        else:
            assert mD is not None or not self.gather_A
            problem_shape_ntile_mnl = (
                None,
                cute.ceil_div(mB.shape[0], self.cta_tile_shape_mnk[1]),
                varlen_args.mCuSeqlensM.shape[0] - 1,
            )
            tile_sched_args = VarlenMTileSchedulerArguments(
                problem_shape_ntile_mnl=problem_shape_ntile_mnl,
                total_m=mD.shape[0] if mD is not None else varlen_args.mAIdx.shape[0],
                cu_seqlens_m=varlen_args.mCuSeqlensM,
                raster_order=scheduler_args.raster_order,
                group_size=scheduler_args.max_swizzle_size,
                tile_shape_mn=self.cta_tile_shape_mnk[:2],
                cluster_shape_mnk=self.cluster_shape_mnk,
                tile_count_semaphore=scheduler_args.tile_count_semaphore,
                is_persistent=self.is_persistent,
            )
        return tile_sched_args

    @cute.jit
    def epi_load_acc_subtile(self, tRS_rAcc: cute.Tensor, tRS_rD: cute.Tensor, epi_idx: int):
        for epi_v in cutlass.range_constexpr(cute.size(tRS_rD)):
            tRS_rD[epi_v] = tRS_rAcc[epi_idx * cute.size(tRS_rD) + epi_v]

    @cute.jit
    def epi_begin(
        self,
        params: EpilogueParams,
        epi_smem_tensors: Tuple[cute.Tensor, ...],
        epi_tile: cute.Tile,
        tiled_copy_t2r: Optional[cute.TiledCopy],
        tiled_copy_r2s: cute.TiledCopy,
        tile_coord_mnkl: cute.Coord,
        varlen_manager: VarlenManager,
        epilogue_barrier: cutlass.pipeline.NamedBarrier,
        tidx: Int32,
    ) -> Tuple[cute.Tensor, ...]:
        return ()

    def epi_begin_loop(
        self, params: EpilogueParams, epi_tensors: Tuple[cute.Tensor, ...], epi_coord: cute.Coord
    ) -> Tuple[cute.Tensor, ...]:
        return ()

    def epi_visit_subtile(
        self,
        params: EpilogueParams,
        epi_loop_tensors: Tuple[cute.Tensor, ...],
        tRS_rD: cute.Tensor,
        tRS_rC: Optional[cute.Tensor] = None,
    ) -> Optional[cute.Tensor]:
        return None

    def epi_visit_acc(
        self,
        params: EpilogueParams,
        acc: cute.Tensor,
        tiled_mma: cute.TiledMma,
        tile_coord_mnkl: cute.Coord,
        tidx: Int32,
    ) -> None:
        pass

    @cute.jit
    def epi_end(
        self,
        params: EpilogueParams,
        epi_tensors: Tuple[cute.Tensor, ...],
        epi_tile: cute.Tile,
        tiled_copy_t2r: Optional[cute.TiledCopy],
        tiled_copy_r2s: cute.TiledCopy,
        tile_coord_mnkl: cute.Coord,
        varlen_manager,
        tidx,
    ) -> None:
        pass

    def epi_to_underlying_arguments(
        self, args: EpilogueArguments, *, loc=None, ip=None
    ) -> EpilogueParams:
        return self.EpilogueParams()

    def epi_get_tma_atoms(
        self, params: EpilogueParams, *, loc=None, ip=None
    ) -> list[cute.CopyAtom]:
        """Subclasses can override this"""
        return []

    def epi_get_tensormap_update_shapes_orders(
        self,
        params: EpilogueParams,
        cu_seqlens_m: cute.Tensor,
        batch_idx: Int32,
        *,
        loc=None,
        ip=None,
    ) -> tuple[list[Int32], list[int]]:
        """Subclasses can override this"""
        return [], []

    @staticmethod
    def epi_smem_bytes_per_stage(
        args: Optional[EpilogueArguments],
        cta_tile_shape_mnk: Tuple[int, int, int],
        epi_tile: cute.Tile,
    ) -> int:
        return 0

    def epi_get_smem_struct(self, params: EpilogueParams):
        return cute.struct.MemRange[Int32, 0]  # Dummy struct

    def epi_get_smem_tensors(self, params: EpilogueParams, storage) -> Tuple[cute.Tensor, ...]:
        return tuple()

    def pingpong_barrier_sync(self, warp_group_idx: Int32, stage: Literal["mma", "epi"]):
        assert stage in ["mma", "epi"]
        barrier = NamedBarrierGemm.MmaWG0 if stage == "mma" else NamedBarrierGemm.EpiWG0
        cute.arch.barrier(
            barrier_id=int(barrier) + warp_group_idx,
            number_of_threads=2 * self.num_threads_per_warp_group,
        )

    def pingpong_barrier_arrive(self, warp_group_idx: Int32, stage: Literal["mma", "epi"]):
        assert stage in ["mma", "epi"]
        barrier = NamedBarrierGemm.MmaWG0 if stage == "mma" else NamedBarrierGemm.EpiWG0
        cute.arch.barrier_arrive(
            barrier_id=int(barrier) + warp_group_idx,
            number_of_threads=2 * self.num_threads_per_warp_group,
        )

    def epilog_smem_copy_atom(self, tiled_mma: cute.TiledMma) -> cute.TiledCopy:
        copy_atom_C = cute.make_copy_atom(
            warp.StMatrix8x8x16bOp(
                self.d_layout.is_m_major_c() if self.d_layout is not None else False,
                num_matrices=4 if self.epi_tile[1] % 16 == 0 else 2,
            ),
            Float16,  # this is just to get the right source layout
        )
        tiled_copy_C_atom = cute.make_tiled_copy_C_atom(copy_atom_C, tiled_mma)
        return tiled_copy_C_atom

    def epilog_smem_store_and_partition(
        self,
        tiled_mma: cute.TiledMma,
        d_layout: Optional[LayoutEnum],
        dtype: Type[cutlass.Numeric],
        sD: Optional[cute.Tensor],
        tidx: Int32,
    ) -> Tuple[cute.TiledCopy, cute.Tensor, cute.Tensor]:
        if d_layout is None:
            d_layout = LayoutEnum.ROW_MAJOR
        tiled_copy_C_atom = self.epilog_smem_copy_atom(tiled_mma)
        # Doesn't work with tile_N % 8 == 0 but tile_n % 16 != since this always
        # get st.matrix with num_matrices=4
        copy_atom_r2s = sm90_utils.sm90_get_smem_store_op(
            d_layout, elem_ty_d=dtype, elem_ty_acc=self.acc_dtype
        )
        tiled_copy_r2s = cute.make_tiled_copy_S(copy_atom_r2s, tiled_copy_C_atom)
        # (R2S, R2S_M, R2S_N, PIPE_D)
        thr_copy_r2s = tiled_copy_r2s.get_slice(tidx)
        tRS_sD = thr_copy_r2s.partition_D(sD) if sD is not None else None
        sD_shape = sD.shape[:2] if sD is not None else self.epi_tile
        tRS_rD_shape = thr_copy_r2s.partition_S(cute.make_identity_tensor(sD_shape)).shape
        tRS_rD = cute.make_fragment(tRS_rD_shape, self.acc_dtype)
        return tiled_copy_r2s, tRS_rD, tRS_sD

    def epilog_smem_load_and_partition(
        self,
        tiled_mma: cute.TiledMma,
        c_layout: LayoutEnum,
        dtype: Type[cutlass.Numeric],
        sC: cute.Tensor,
        tRS_rD_layout: cutlass.Layout,
        tidx: Int32,
    ) -> Tuple[cute.TiledCopy, cute.Tensor, cute.Tensor]:
        tiled_copy_C_atom = self.epilog_smem_copy_atom(tiled_mma)
        copy_atom_s2r = copy_utils.sm90_get_smem_load_op(c_layout, dtype)
        tiled_copy_s2r = cute.make_tiled_copy_S(copy_atom_s2r, tiled_copy_C_atom)
        thr_copy_s2r = tiled_copy_s2r.get_slice(tidx)
        tSR_sC = thr_copy_s2r.partition_S(sC)
        tRS_rC = cute.make_fragment(tRS_rD_layout, dtype)
        tSR_rC = thr_copy_s2r.retile(tRS_rC)
        return tiled_copy_s2r, tRS_rC, tSR_rC, tSR_sC

    def epilog_gmem_copy_and_partition(
        self,
        atom: Union[cute.CopyAtom, cute.TiledCopy],
        mD_mn: cute.Tensor,
        tile_shape_mn: cute.Tile,
        epi_tile: cute.Tile,
        sD: cute.Tensor,
        tile_coord_mnkl: cute.Coord,
        tma_desc_ptr: Optional[cute.Pointer] = None,
    ) -> Tuple[cute.Tensor, cute.Tensor]:
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
            tma_desc_ptr=tma_desc_ptr,
        )

    def make_ab_pipeline(
        self,
        tiled_mma: cute.TiledMma,
        cluster_layout_vmnk: cute.Layout,
        ab_pipeline_mbar_ptr: cute.Pointer,
    ):
        # Threads/warps participating in this pipeline
        producer_cnt = 1 if const_expr(not self.gather_A) else 1 + self.num_ab_load_warps * 32
        ab_pipeline_producer_group = pipeline.CooperativeGroup(pipeline.Agent.Thread, producer_cnt)
        # Each warp will contribute to the arrive count with the number of mcast size
        mcast_size = self.num_mcast_ctas_a + self.num_mcast_ctas_b - 1
        consumer_arrive_cnt = mcast_size * tiled_mma.size // cute.arch.WARP_SIZE
        ab_pipeline_consumer_group = pipeline.CooperativeGroup(
            pipeline.Agent.Thread, consumer_arrive_cnt
        )
        pipeline_cls = pipeline.PipelineTmaAsync if not self.gather_A else PipelineTmaCpAsync
        return pipeline_cls.create(
            barrier_storage=ab_pipeline_mbar_ptr,
            num_stages=self.ab_stage,
            producer_group=ab_pipeline_producer_group,
            consumer_group=ab_pipeline_consumer_group,
            tx_count=self.num_tma_load_bytes,
            cta_layout_vmnk=cluster_layout_vmnk,
        )

    def make_epi_pipeline(
        self, c_smem_layout: cute.Layout | cute.ComposedLayout, epi_pipeline_mbar_ptr: cute.Pointer
    ):
        # Threads/warps participating in this pipeline
        epi_pipeline_producer_group = pipeline.CooperativeGroup(pipeline.Agent.Thread)
        # Each warp will contribute 1 to the arrive count
        consumer_arrive_cnt = self.num_epi_warps
        epi_pipeline_consumer_group = pipeline.CooperativeGroup(
            pipeline.Agent.Thread, consumer_arrive_cnt
        )
        tma_copy_c_bytes = cute.size_in_bytes(self.c_dtype, c_smem_layout)
        return pipeline.PipelineTmaAsync.create(
            barrier_storage=epi_pipeline_mbar_ptr,
            num_stages=self.epi_c_stage,
            producer_group=epi_pipeline_producer_group,
            consumer_group=epi_pipeline_consumer_group,
            tx_count=tma_copy_c_bytes,
        )

    def make_epi_store_pipeline(self):
        # Threads/warps participating in tma store pipeline
        num_epi_threads = self.num_epi_warps * cute.arch.WARP_SIZE
        epi_store_producer_group = pipeline.CooperativeGroup(pipeline.Agent.Thread, num_epi_threads)
        return pipeline.PipelineTmaStore.create(
            num_stages=self.epi_stage, producer_group=epi_store_producer_group
        )

    def make_sched_pipeline(
        self, cluster_layout_mnk: cute.Layout, sched_pipeline_mbar_ptr: cute.Pointer, varlen_k: bool
    ):
        # Threads/warps participating in this pipeline
        sched_pipeline_producer_group = pipeline.CooperativeGroup(pipeline.Agent.Thread)
        cluster_size = cute.size(cluster_layout_mnk)
        # Each warp that are not the scheduler warp will contribute 1 to the arrive count
        # If pingpong and varlen_k, then all 8 mma warps will participate in the scheduler barrier
        # at each round. If pingpong and not varlen_k, then only 4 mma warp will participate.
        consumer_arrive_cnt = (
            (self.mma_warp_groups if not (self.pingpong and not varlen_k) else 1) * 4
            + self.num_ab_load_warps
        ) * cluster_size - 1
        sched_pipeline_consumer_group = pipeline.CooperativeGroup(
            pipeline.Agent.Thread, consumer_arrive_cnt
        )
        return pipeline.PipelineAsync.create(
            barrier_storage=sched_pipeline_mbar_ptr,
            num_stages=self.sched_stage,
            producer_group=sched_pipeline_producer_group,
            consumer_group=sched_pipeline_consumer_group,
            # If there's cluster, the consumers must arrive at the mbar of CTA 0 in the cluster.
            consumer_mask=None if const_expr(cluster_size == 1) else 0,
        )

    @classmethod
    def _compute_stages(
        cls,
        cta_tile_shape_mnk: Tuple[int, int, int],
        epi_tile: Tuple[int, int],
        a_dtype: Type[cutlass.Numeric],
        b_dtype: Type[cutlass.Numeric],
        d_dtype: Optional[Type[cutlass.Numeric]],
        c_dtype: Optional[Type[cutlass.Numeric]],
        epilogue_args: EpilogueArguments,
        smem_capacity: int,
        occupancy: int,
        overlap_sD_sA: bool = False,
    ) -> Tuple[int, int]:
        """Computes the number of stages for A/B/C operands based on heuristics.

        :param cta_tile_shape_mnk: The shape (M, N, K) of the CTA tile.
        :type cta_tile_shape_mnk: Tuple[int, int, int]
        :param a_dtype: Data type of operand A.
        :type a_dtype: type[cutlass.Numeric]
        :param b_dtype: Data type of operand B.
        :type b_dtype: type[cutlass.Numeric]
        :param smem_capacity: Total available shared memory capacity in bytes.
        :type smem_capacity: int
        :param occupancy: Target number of CTAs per SM (occupancy).
        :type occupancy: int

        :return: A tuple containing the computed number of stages for:
                 (A/B operand stages, epilogue stages)
        :rtype: Tuple[int, int]
        """

        epi_stage = 4 if epi_tile[1] <= 16 else 2
        if overlap_sD_sA:
            epi_bytes = 0
        else:
            d_bytes_per_stage = (
                cute.size(epi_tile) * d_dtype.width // 8 if d_dtype is not None else 0
            )
            epi_bytes_per_stage = d_bytes_per_stage + cls.epi_smem_bytes_per_stage(
                epilogue_args, cta_tile_shape_mnk, epi_tile
            )
            epi_bytes = epi_bytes_per_stage * epi_stage
        epi_c_stage = 0 if c_dtype is None else (4 if epi_tile[1] <= 16 else 2)
        if c_dtype is not None:
            epi_bytes += cute.size(epi_tile) * c_dtype.width // 8 * epi_c_stage

        a_shape = cute.slice_(cta_tile_shape_mnk, (None, 0, None))
        b_shape = cute.slice_(cta_tile_shape_mnk, (0, None, None))
        ab_bytes_per_stage = (
            cute.size(a_shape) * a_dtype.width // 8 + cute.size(b_shape) * b_dtype.width // 8
        )
        mbar_helpers_bytes = 1024

        remaining_bytes = smem_capacity // occupancy - mbar_helpers_bytes - epi_bytes
        ab_stage = remaining_bytes // ab_bytes_per_stage

        # Refine epilogue stages:
        # Calculate remaining smem after allocating for A/B stages and reserved bytes
        # Add remaining unused smem to epilogue
        if not overlap_sD_sA and epi_bytes_per_stage > 0:
            epi_stage += (remaining_bytes - ab_bytes_per_stage * ab_stage) // epi_bytes_per_stage
        return ab_stage, epi_stage, epi_c_stage

    @staticmethod
    def _sm90_compute_tile_shape_or_override(
        cta_tile_shape_mnk: Tuple[int, int, int],
        atom_layout_mnk: Tuple[int, int, int],
        element_type: Optional[Type[cutlass.Numeric]] = None,
        epi_tile_override: Tuple[int, int] | None = None,
    ) -> Tuple[int, int]:
        """Compute the epilogue tile shape or use override if provided.

        :param cta_tile_shape_mnk: CTA tile shape (M,N,K)
        :type cta_tile_shape_mnk: Tuple[int, int, int]
        :param element_type: Data type of elements
        :type element_type: type[cutlass.Numeric]
        :param is_cooperative: Whether to use cooperative approach
        :type is_cooperative: bool
        :param epi_tile_override: Optional override for epilogue tile shape
        :type epi_tile_override: Tuple[int, int] or None

        :return: Computed epilogue tile shape
        :rtype: Tuple[int, int]
        """
        if epi_tile_override is not None:
            return epi_tile_override
        if cta_tile_shape_mnk[0] % 128 == 0 and atom_layout_mnk[0] > 1:
            tile_m = math.gcd(128, cute.size(cta_tile_shape_mnk, mode=[0]))
            tile_n = math.gcd(32, cute.size(cta_tile_shape_mnk, mode=[1]))
        elif cta_tile_shape_mnk[0] % 192 == 0 and atom_layout_mnk[0] > 1:
            tile_m = math.gcd(192, cute.size(cta_tile_shape_mnk, mode=[0]))
            tile_n = math.gcd(32, cute.size(cta_tile_shape_mnk, mode=[1]))
        else:
            # In the case of tile shape 128 x N but atom_layout 1 x 2, we need to set
            # epi_tile_m = 64. If epi_tile_m = 128, the epilogue would iterate along the
            # M dimension first, then move to the N dimension. But the accumulator in registers
            # iterate along the N dimension first, then move to the M dimension.
            # We could change the epilogue to accommodate this,
            # but it's easier to just set epi_tile_m = 64.
            n_perf = 64 if element_type is not None and element_type.width == 8 else 32
            tile_m = math.gcd(64, cute.size(cta_tile_shape_mnk, mode=[0]))
            tile_n = math.gcd(n_perf, cute.size(cta_tile_shape_mnk, mode=[1]))
        return (tile_m, tile_n)

    @staticmethod
    def _make_smem_layouts(
        cta_tile_shape_mnk: Tuple[int, int, int],
        epi_tile: Tuple[int, int],
        a_dtype: Type[cutlass.Numeric],
        a_layout: LayoutEnum,
        b_dtype: Type[cutlass.Numeric],
        b_layout: LayoutEnum,
        ab_stage: int,
        d_dtype: Optional[Type[cutlass.Numeric]],
        d_layout: LayoutEnum,
        epi_stage: int,
        c_dtype: Optional[Type[cutlass.Numeric]],
        c_layout: Optional[LayoutEnum],
        epi_c_stage: int,
    ) -> Tuple[
        cute.ComposedLayout, cute.ComposedLayout, cute.ComposedLayout, Optional[cute.ComposedLayout]
    ]:
        """Create shared memory layouts for A, B, and C tensors.

        :param cta_tile_shape_mnk: CTA tile shape (M,N,K)
        :type cta_tile_shape_mnk: Tuple[int, int, int]
        :param epi_tile: Epilogue tile shape
        :type epi_tile: Tuple[int, int]
        :param a_dtype: Data type for matrix A
        :type a_dtype: type[cutlass.Numeric]
        :param a_layout: Layout enum for matrix A
        :type a_layout: LayoutEnum
        :param b_dtype: Data type for matrix B
        :type b_dtype: type[cutlass.Numeric]
        :param b_layout: Layout enum for matrix B
        :type b_layout: LayoutEnum
        :param ab_stage: Number of stages for A/B tensors
        :type ab_stage: int
        :param d_dtype: Data type for output matrix D
        :type d_dtype: type[cutlass.Numeric]
        :param d_layout: Layout enum for the output matrix C
        :type d_layout: LayoutEnum
        :param epi_stage: Number of epilogue stages
        :type epi_stage: int

        :return: Tuple of shared memory layouts for A, B, and C
        :rtype: Tuple[cute.ComposedLayout, cute.ComposedLayout, cute.ComposedLayout]
        """
        a_smem_shape = cute.slice_(cta_tile_shape_mnk, (None, 0, None))

        a_is_k_major = a_layout.sm90_mma_major_mode() == warpgroup.OperandMajorMode.K
        b_is_k_major = b_layout.sm90_mma_major_mode() == warpgroup.OperandMajorMode.K
        a_major_mode_size = cta_tile_shape_mnk[2 if a_is_k_major else 0]
        a_smem_layout_atom = warpgroup.make_smem_layout_atom(
            sm90_utils.get_smem_layout_atom(a_layout, a_dtype, a_major_mode_size),
            a_dtype,
        )
        a_smem_layout_staged = cute.tile_to_shape(
            a_smem_layout_atom,
            cute.append(a_smem_shape, ab_stage),
            order=(0, 1, 2) if a_is_k_major else (1, 0, 2),
        )

        b_smem_shape = cute.slice_(cta_tile_shape_mnk, (0, None, None))

        b_major_mode_size = cta_tile_shape_mnk[2 if b_is_k_major else 1]
        b_smem_layout_atom = warpgroup.make_smem_layout_atom(
            sm90_utils.get_smem_layout_atom(b_layout, b_dtype, b_major_mode_size),
            b_dtype,
        )
        b_smem_layout_staged = cute.tile_to_shape(
            b_smem_layout_atom,
            cute.append(b_smem_shape, ab_stage),
            order=(0, 1, 2) if b_is_k_major else (1, 0, 2),
        )

        epi_smem_layout_staged = None
        assert d_dtype is None

        epi_c_smem_layout_staged = None
        assert c_dtype is None

        return (
            a_smem_layout_staged,
            b_smem_layout_staged,
            epi_smem_layout_staged,
            epi_c_smem_layout_staged,
        )

    @staticmethod
    def _make_tma_epi_atoms_and_tensors(
        tensor_d: cute.Tensor,
        epi_smem_layout_staged: cute.ComposedLayout,
        epi_tile: Tuple[int, int],
        op_type: Literal["store", "load", "add"],
    ) -> Tuple[cute.CopyAtom, cute.Tensor]:
        """Create TMA atoms and tensors for storing D or loading C.

        :param tensor_d: Output tensor D
        :type tensor_d: cute.Tensor
        :param epi_smem_layout_staged: Shared memory layout for epilogue
        :type epi_smem_layout_staged: cute.ComposedLayout
        :param epi_tile: Epilogue tile shape
        :type epi_tile: Tuple[int, int]

        :return: TMA atom and tensor for C
        :rtype: Tuple[cute.CopyAtom, cute.Tensor]
        """
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

    @staticmethod
    def _make_tma_atoms_and_tensors(
        tensor: cute.Tensor,
        smem_layout: cute.ComposedLayout,
        smem_tile: Tuple[int, int],
        mcast_dim: int,
    ) -> Tuple[cute.CopyAtom, cute.Tensor]:
        """Create TMA atoms and tensors for input tensors.

        :param tensor: Input tensor (A or B)
        :type tensor: cute.Tensor
        :param smem_layout: Shared memory layout for the tensor
        :type smem_layout: cute.ComposedLayout
        :param smem_tile: Shared memory tile shape
        :type smem_tile: Tuple[int, int]
        :param mcast_dim: Multicast dimension
        :type mcast_dim: int

        :return: TMA atom and tensor
        :rtype: Tuple[cute.CopyAtom, cute.Tensor]
        """
        op = (
            cpasync.CopyBulkTensorTileG2SOp()
            if mcast_dim == 1
            else cpasync.CopyBulkTensorTileG2SMulticastOp()
        )
        tma_atom, tma_tensor = cpasync.make_tiled_tma_atom(
            op,
            tensor,
            smem_layout,
            smem_tile,
            num_multicast=mcast_dim,
        )
        return tma_atom, tma_tensor

    def _make_gmem_tiled_copy_A(self, dtype, major_mode, num_threads, copy_bits=128):
        atom_async_copy = cute.make_copy_atom(
            cpasync.CopyG2SOp(cache_mode=cpasync.LoadCacheMode.GLOBAL),
            dtype,
            num_bits_per_copy=copy_bits,
        )
        copy_elems = copy_bits // dtype.width
        loads_per_cache_line = 128 * 8 // copy_bits  # 128 bytes per cache line
        shape_dim_1 = cute.size(self.cta_tile_shape_mnk[2]) // copy_elems
        if shape_dim_1 > loads_per_cache_line:
            shape_dim_1 = math.gcd(shape_dim_1, loads_per_cache_line)
        # thread layout for copy
        thread_layout = cute.make_layout(
            (num_threads // shape_dim_1, shape_dim_1), stride=(shape_dim_1, 1)
        )
        if major_mode != LayoutEnum.ROW_MAJOR:
            shape_dim_0 = cute.size(self.cta_tile_shape_mnk[0]) // copy_elems
            if shape_dim_0 > loads_per_cache_line:
                shape_dim_0 = math.gcd(shape_dim_0, loads_per_cache_line)
            thread_layout = cute.make_layout(
                (shape_dim_0, num_threads // shape_dim_0), stride=(1, shape_dim_0)
            )
        # Value layout for copy
        value_layout = (
            cute.make_layout((1, copy_elems))
            if major_mode == LayoutEnum.ROW_MAJOR
            else cute.make_layout((copy_elems, 1))
        )
        return cute.make_tiled_copy_tv(atom_async_copy, thread_layout, value_layout)

    @staticmethod
    def is_valid_dtypes(
        a_dtype: Type[cutlass.Numeric],
        b_dtype: Type[cutlass.Numeric],
        acc_dtype: Type[cutlass.Numeric],
        d_dtype: Optional[Type[cutlass.Numeric]],
        a_major: str,
        b_major: str,
    ) -> bool:
        """
        Check if the dtypes are valid

        :param a_dtype: The data type of tensor A
        :type a_dtype: Type[cutlass.Numeric]
        :param b_dtype: The data type of tensor B
        :type b_dtype: Type[cutlass.Numeric]
        :param acc_dtype: The data type of the accumulator
        :type acc_dtype: Type[cutlass.Numeric]
        :param d_dtype: The data type of the output tensor
        :type d_dtype: Type[cutlass.Numeric]
        :param a_major: major mode of tensor A
        :type a_major: str
        :param b_major: major mode of tensor B
        :type b_major: str

        :return: True if the dtypes are valid, False otherwise
        :rtype: bool
        """
        is_valid = True
        if a_dtype not in {
            Float16,
            cutlass.BFloat16,
            cutlass.Float8E4M3FN,
            cutlass.Float8E5M2,
        }:
            is_valid = False
        # tested b_dtype
        if b_dtype not in {
            Float16,
            cutlass.BFloat16,
            cutlass.Float8E4M3FN,
            cutlass.Float8E5M2,
        }:
            is_valid = False
        if acc_dtype not in {Float32, Float16}:
            is_valid = False
        # tested d_dtype
        if d_dtype not in {
            None,
            Float32,
            Float16,
            cutlass.BFloat16,
            cutlass.Float8E4M3FN,
            cutlass.Float8E5M2,
        }:
            is_valid = False
        # make sure a_dtype == b_dtype for Float16
        if a_dtype.width == 16 and a_dtype != b_dtype:
            is_valid = False
        # make sure a_dtype.width == b_dtype.width (i.e, Float8E4M3FN or Float8E5M2)
        if a_dtype.width != b_dtype.width:
            is_valid = False

        # for Float8 types, this implementation only supports k-major layout
        if (a_dtype.width == 8 and a_major != "k") or (b_dtype.width == 8 and b_major != "k"):
            is_valid = False
        return is_valid

"""
A high-performance persistent batched dense GEMM example for the NVIDIA Blackwell SM100 architecture
using CUTE DSL.
- Matrix A is MxKxL, L is batch dimension, A can be row-major("K") or column-major("M")
- Matrix B is NxKxL, L is batch dimension, B can be row-major("N") or column-major("K")
- Matrix C is MxNxL, L is batch dimension, C can be row-major("N") or column-major("M")

This GEMM kernel supports the following features:
    - Utilizes Tensor Memory Access (TMA) for efficient memory operations
    - Utilizes Blackwell's tcgen05.mma for matrix multiply-accumulate (MMA) operations (including 2cta mma instructions)
    - Implements TMA multicast with cluster to reduce L2 memory traffic
    - Support persistent tile scheduling to better overlap memory load/store with mma between tiles
    - Support warp specialization to avoid explicit pipelining between mainloop load and mma

This GEMM works as follows:
1. DMA warp: Load A and B matrices from global memory (GMEM) to shared memory (SMEM) using TMA operations.
2. MMA warp: Perform matrix multiply-accumulate (MMA) operations using tcgen05.mma instruction.
3. EPILOGUE warp:
    - Load completed accumulator from tensor memory (TMEM) to registers (RMEM) using tcgen05.ld.
    - Type convert C matrix to output type.
    - Optionally store C matrix from registers (RMEM) to shared memory (SMEM) to global memory (GMEM) with TMA operations,
      or directly store C matrix from registers (RMEM) to global memory (GMEM) without TMA operations.

SM100 tcgen05.mma instructions operate as follows:
- Read matrix A from SMEM
- Read matrix B from SMEM
- Write accumulator to TMEM
The accumulator in TMEM must then be loaded to registers before writing back to GMEM.

Input arguments to this example is same as dense_gemm.py.

.. code-block:: bash

    python examples/blackwell/dense_gemm_persistent.py                          \
      --ab_dtype Float16 --d_dtype Float16 --acc_dtype Float32                  \
      --mma_tiler_mn 256,128 --cluster_shape_mn 2,1                             \
      --mnkl 8192,8192,8192,1                                                   \
      --use_2cta_instrs

To collect performance with NCU profiler:

.. code-block:: bash

    ncu python examples/blackwell/dense_gemm_persistent.py                     \
      --ab_dtype Float16 --d_dtype Float16 --acc_dtype Float32                 \
      --mma_tiler_mn 256,128 --cluster_shape_mn 2,1                            \
      --mnkl 8192,8192,8192,1                                                  \
      --use_2cta_instrs                                        \
      --warmup_iterations 1 --iterations 10 --skip_ref_check


Constraints are same as dense_gemm.py:
* Supported input data types: fp16, bf16, tf32, int8, uint8, fp8 (e4m3fn, e5m2),
  see detailed valid dtype combinations in below GemmSm100 class documentation
* A/B tensor must have the same data type
* Mma tiler M must be 64/128 (use_2cta_instrs=False) or 128/256 (use_2cta_instrs=True)
* Mma tiler N must be 32-256, step 32
* Cluster shape M/N must be positive and power of 2, total cluster size <= 16
* Cluster shape M must be multiple of 2 if use_2cta_instrs=True
* The contiguous dimension of A/B/C tensors must be at least 16 bytes aligned,
  i.e, number of elements is a multiple of 4, 8, and 16 for TFloat32,
  Float16/BFloat16, and Int8/Uint8/Float8, respectively.
* OOB tiles are not allowed when TMA store is disabled
"""


class GemmSm100(GemmSm90):
    """This class implements batched matrix multiplication (C = A x B) with support for various data types
    and architectural features specific to Blackwell GPUs with persistent tile scheduling and warp specialization.

    :param acc_dtype: Data type for accumulation during computation
    :type acc_dtype: type[cutlass.Numeric]
    :param mma_tiler_mn: Shape of the Matrix Multiply-Accumulate (MMA) tile (M,N)
    :type mma_tiler_mn: Tuple[int, int]
    :param cluster_shape_mn: Cluster dimensions (M,N) for parallel processing
    :type cluster_shape_mn: Tuple[int, int]

    :note: In current version, A and B tensor must have the same data type
        - i.e., Float8E4M3FN for A and Float8E5M2 for B is not supported

    :note: Supported A/B data types:
        - TFloat32
        - Float16/BFloat16
        - Int8/Uint8
        - Float8E4M3FN/Float8E5M2

    :note: Supported accumulator data types:
        - Float32 (for all floating point A/B data types)
        - Float16 (only for fp16 and fp8 A/B data types)
        - Int32 (only for uint8/int8 A/B data types)

    :note: Supported C data types:
        - Float32 (for float32 and int32 accumulator data types)
        - Int32 (for float32 and int32 accumulator data types)
        - Float16/BFloat16 (for fp16 and fp8 accumulator data types)
        - Int8/Uint8 (for uint8/int8 accumulator data types)
        - Float8E4M3FN/Float8E5M2 (for float32 accumulator data types)

    :note: Constraints:
        - MMA tiler M must be 64/128 (use_2cta_instrs=False) or 128/256 (use_2cta_instrs=True)
        - MMA tiler N must be 32-256, step 32
        - Cluster shape M must be multiple of 2 if use_2cta_instrs=True
        - Cluster shape M/N must be positive and power of 2, total cluster size <= 16

    Example:
        >>> gemm = GemmSm100(
        ...     acc_dtype=Float32,
        ...     mma_tiler_mn=(128, 128),
        ...     cluster_shape_mn=(2, 2)
        ... )
        >>> gemm(mA, mB, mD, max_active_clusters, stream)
    """

    arch = 100
    num_epi_tensormaps = GemmSm90.num_epi_tensormaps

    EpilogueArguments = GemmSm90.EpilogueArguments
    EpilogueParams = GemmSm90.EpilogueParams

    def __init__(
        self,
        acc_dtype: Type[cutlass.Numeric],
        a_dtype: Type[cutlass.Numeric],  # ignored for now
        mma_tiler_mn: Tuple[int, int],
        cluster_shape_mnk: Tuple[int, int, int],
        sf_vec_size: Optional[int] = None,
        gather_A: bool = False,
    ):
        """Initializes the configuration for a Blackwell dense GEMM kernel.

        This configuration includes several key aspects:

        1.  MMA Instruction Settings (tcgen05):
            - acc_dtype: Data types for MMA accumulator.
            - mma_tiler_mn: The (M, N) shape of the MMA instruction tiler.
            - use_2cta_instrs: Boolean indicating if the tcgen05 MMA variant
              with cta_group=2 should be used.

        2.  Cluster Shape:
            - cluster_shape_mnk: The (ClusterM, ClusterN) shape of the CTA cluster.

        :param acc_dtype: Data type of the accumulator.
        :type acc_dtype: type[cutlass.Numeric]
        :param mma_tiler_mn: Tuple (M, N) shape of the MMA instruction.
        :type mma_tiler_mn: Tuple[int, int]
        :param cluster_shape_mnk: Tuple (ClusterM, ClusterN) shape of the cluster.
        :type cluster_shape_mnk: Tuple[int, int]
        """

        self.acc_dtype: Type[cutlass.Numeric] = acc_dtype
        self.use_2cta_instrs = cluster_shape_mnk[0] == 2 and mma_tiler_mn[0] in (256,)
        self.cluster_shape_mnk = cluster_shape_mnk
        assert cluster_shape_mnk[2] == 1, "Cluster shape K must be 1"
        # K dimension is deferred in _setup_attributes
        self.mma_tiler = (*mma_tiler_mn, 1)
        self.sf_vec_size = sf_vec_size
        self.blockscaled = sf_vec_size is not None
        self.is_persistent = True
        self.pingpong = False  # for compatibility with GemmSm90
        self.gather_A = gather_A
        if gather_A:
            assert cluster_shape_mnk[1] == 1, "Cluster shape N must be 1 for gather A "

        self.cta_group = tcgen05.CtaGroup.TWO if self.use_2cta_instrs else tcgen05.CtaGroup.ONE

        self.num_ab_load_warps = 1 if not self.gather_A else 5
        self.occupancy = 1
        # Set specialized warp ids
        self.epilog_warp_id = (0, 1, 2, 3)
        self.mma_warp_id = 4
        self.ab_load_warp_id = 5
        self.epi_load_warp_id = self.ab_load_warp_id + self.num_ab_load_warps
        self.scheduler_warp_id = self.epi_load_warp_id + 1
        self.num_epi_warps = len(self.epilog_warp_id)
        self.threads_per_cta = cute.arch.WARP_SIZE * (
            self.num_ab_load_warps
            + len(
                (
                    self.mma_warp_id,
                    self.epi_load_warp_id,
                    self.scheduler_warp_id,
                    *self.epilog_warp_id,
                )
            )
        )

    def _setup_attributes(self, epilogue_args: EpilogueArguments, varlen_args: VarlenArguments):
        """Set up configurations that are dependent on GEMM inputs

        This method configures various attributes based on the input tensor properties
        (data types, leading dimensions) and kernel settings:
        - Configuring tiled MMA
        - Computing MMA/cluster/tile shapes
        - Computing cluster layout
        - Computing multicast CTAs for A/B
        - Computing epilogue subtile
        - Setting up A/B/C stage counts in shared memory
        - Computing A/B/C shared memory layout
        - Computing tensor memory allocation columns
        """
        # Compute mma instruction shapes
        mma_inst_bits_k = 256
        # (MMA_Tile_Shape_M, MMA_Tile_Shape_N, MMA_Inst_Shape_K)
        self.mma_inst_shape_mnk = (
            self.mma_tiler[0],
            self.mma_tiler[1],
            mma_inst_bits_k // self.a_dtype.width,
        )
        # (CTA_Tile_Shape_M, Round_Up(MMA_Tile_Shape_N, 128), MMA_Inst_Shape_K)
        self.mma_inst_shape_mnk_sfb = (
            self.mma_inst_shape_mnk[0] // (2 if self.use_2cta_instrs else 1),
            cute.round_up(self.mma_inst_shape_mnk[1], 128),
            self.mma_inst_shape_mnk[2],
        )

        # Configure tiled mma
        if const_expr(not self.blockscaled):
            self.tiled_mma = sm100_utils.make_trivial_tiled_mma(
                self.a_dtype,
                self.a_major_mode,
                self.b_major_mode,
                self.acc_dtype,
                self.cta_group,
                self.mma_tiler[:2],
            )
            self.tiled_mma_sfb = None
        else:
            self.tiled_mma = sm100_utils.make_blockscaled_trivial_tiled_mma(
                self.a_dtype,
                self.a_major_mode,
                self.b_major_mode,
                self.sf_dtype,
                self.sf_vec_size,
                self.cta_group,
                self.mma_inst_shape_mnk[:2],
            )
            self.tiled_mma_sfb = sm100_utils.make_blockscaled_trivial_tiled_mma(
                self.a_dtype,
                self.a_major_mode,
                self.b_major_mode,
                self.sf_dtype,
                self.sf_vec_size,
                tcgen05.CtaGroup.ONE,
                self.mma_inst_shape_mnk_sfb[:2],
            )

        # Compute mma/cluster/tile shapes
        mma_inst_tile_k = 4
        self.mma_tiler = (
            self.mma_inst_shape_mnk[0],
            self.mma_inst_shape_mnk[1],
            self.mma_inst_shape_mnk[2] * mma_inst_tile_k,
        )
        if const_expr(self.blockscaled):
            self.mma_tiler_sfb = (
                self.mma_inst_shape_mnk_sfb[0],
                self.mma_inst_shape_mnk_sfb[1],
                self.mma_inst_shape_mnk_sfb[2] * mma_inst_tile_k,
            )
        else:
            self.mma_tiler_sfb = None
        self.cta_tile_shape_mnk = (
            self.mma_tiler[0] // cute.size(self.tiled_mma.thr_id.shape),
            self.mma_tiler[1],
            self.mma_tiler[2],
        )

        # Compute cluster layout
        self.cluster_layout_vmnk = cute.tiled_divide(
            cute.make_layout(self.cluster_shape_mnk),
            (self.tiled_mma.thr_id.shape,),
        )
        if const_expr(self.blockscaled):
            self.cluster_layout_sfb_vmnk = cute.tiled_divide(
                cute.make_layout(self.cluster_shape_mnk),
                (self.tiled_mma_sfb.thr_id.shape,),
            )
        else:
            self.cluster_layout_sfb_vmnk = None

        # Compute number of multicast CTAs for A/B
        self.num_mcast_ctas_a = cute.size(self.cluster_layout_vmnk.shape[2])
        if self.gather_A:
            assert self.num_mcast_ctas_a == 1
        self.num_mcast_ctas_b = cute.size(self.cluster_layout_vmnk.shape[1])
        self.is_a_mcast = self.num_mcast_ctas_a > 1
        self.is_b_mcast = self.num_mcast_ctas_b > 1
        if const_expr(self.blockscaled):
            self.num_mcast_ctas_sfb = cute.size(self.cluster_layout_sfb_vmnk.shape[1])
            self.is_sfb_mcast = self.num_mcast_ctas_sfb > 1

        # Compute epilogue subtile
        self.epi_tile = sm100_utils.compute_epilogue_tile_shape(
            self.cta_tile_shape_mnk,
            self.use_2cta_instrs,
            self.d_layout if self.d_layout is not None else LayoutEnum.ROW_MAJOR,
            self.d_dtype if self.d_dtype is not None else cutlass.BFloat16,
            layout_c=self.c_layout,
            elem_ty_c=self.c_dtype,
        )

        # Setup A/B/C stage count in shared memory and ACC stage count in tensor memory
        prefetch_A_idx = (
            None
            if not self.gather_A
            else ("varlen_m" if varlen_args.mCuSeqlensM is not None else "varlen_k")
        )
        (
            self.num_acc_stage,
            self.ab_stage,
            self.epi_stage,
            self.epi_c_stage,
        ) = self._compute_stages(
            self.tiled_mma,
            self.mma_tiler,
            self.cta_tile_shape_mnk,
            self.epi_tile,
            self.a_dtype,
            self.b_dtype,
            self.sf_dtype,
            self.sf_vec_size,
            self.d_dtype,
            self.c_dtype,
            self.d_layout,
            self.c_layout,
            epilogue_args,
            prefetch_A_idx,
            cutlass.utils.get_smem_capacity_in_bytes(f"sm_{self.arch}"),  # smem_capacity
            self.occupancy,
        )
        self.sched_stage = 1
        self.a_prefetch_stage = (
            0
            if not self.gather_A
            else (2 if varlen_args.mCuSeqlensM is not None else self.ab_stage)
        )

        # Compute A/B/SFA/SFB/C shared memory layout
        self.a_smem_layout_staged = sm100_utils.make_smem_layout_a(
            self.tiled_mma, self.mma_tiler, self.a_dtype, self.ab_stage
        )
        self.a_smem_load_layout_staged = self.a_smem_layout_staged
        assert not const_expr(self.gather_A)
        self.b_smem_layout_staged = sm100_utils.make_smem_layout_b(
            self.tiled_mma, self.mma_tiler, self.b_dtype, self.ab_stage
        )
        self.epi_smem_layout_staged = None
        if const_expr(self.d_dtype is not None):
            self.epi_smem_layout_staged = sm100_utils.make_smem_layout_epi(
                self.d_dtype, self.d_layout, self.epi_tile, self.epi_stage
            )
        self.epi_c_smem_layout_staged = None
        if const_expr(self.c_dtype is not None):
            self.epi_c_smem_layout_staged = sm100_utils.make_smem_layout_epi(
                self.c_dtype, self.c_layout, self.epi_tile, self.epi_c_stage
            )
        if const_expr(self.blockscaled):
            self.sfa_smem_layout_staged = blockscaled_utils.make_smem_layout_sfa(
                self.tiled_mma,
                self.mma_tiler,
                self.sf_vec_size,
                self.ab_stage,
            )
            self.sfb_smem_layout_staged = blockscaled_utils.make_smem_layout_sfb(
                self.tiled_mma,
                self.mma_tiler,
                self.sf_vec_size,
                self.ab_stage,
            )
        else:
            self.sfa_smem_layout_staged, self.sfb_smem_layout_staged = None, None

        # Compute the number of tensor memory allocation columns
        if const_expr(not self.blockscaled):
            self.num_tmem_alloc_cols = self._compute_num_tmem_alloc_cols(
                self.tiled_mma, self.mma_tiler, self.num_acc_stage
            )
        else:
            SM100_TMEM_CAPACITY_COLUMNS = 512
            self.num_tmem_alloc_cols = SM100_TMEM_CAPACITY_COLUMNS

    @cute.jit
    def __call__(
        self,
        mA: cute.Tensor,
        mB: cute.Tensor,
        mD: Optional[cute.Tensor],
        mC: Optional[cute.Tensor],
        epilogue_args: ArgumentsBase,
        scheduler_args: TileSchedulerOptions,
        varlen_args: Optional[VarlenArguments],
        stream: cuda.CUstream,
        mSFA: Optional[cute.Tensor] = None,
        mSFB: Optional[cute.Tensor] = None,
    ):
        """Execute the GEMM operation in steps:
        - Setup static attributes before smem/grid/tma computation
        - Setup TMA load/store atoms and tensors
        - Compute grid size with regard to hardware constraints
        - Define shared storage for kernel
        - Launch the kernel synchronously

        :param mA: Input tensor A
        :type mA: cute.Tensor
        :param mB: Input tensor B
        :type mB: cute.Tensor
        :param mD: Output tensor D
        :type mD: cute.Tensor
        :param max_active_clusters: Maximum number of active clusters
        :type max_active_clusters: cutlass.Constexpr
        :param stream: CUDA stream for asynchronous execution
        :type stream: cuda.CUstream
        :raises TypeError: If input data types are incompatible with the MMA instruction.
        :raises AssertionError: If OOB (Out-Of-Bounds) tiles are present when TMA store is disabled.
        """
        if const_expr(self.blockscaled):
            assert mSFA is not None and mSFB is not None
        # Setup static attributes before smem/grid/tma computation
        self.a_dtype = mA.element_type
        self.b_dtype = mB.element_type
        self.d_dtype = mD.element_type if mD is not None else None
        self.c_dtype = mC.element_type if mC is not None else None
        self.sf_dtype: Optional[Type[cutlass.Numeric]] = (
            mSFA.element_type if mSFA is not None else None
        )
        self.a_layout = LayoutEnum.from_tensor(mA)
        self.b_layout = LayoutEnum.from_tensor(mB)
        self.d_layout = LayoutEnum.from_tensor(mD) if mD is not None else None
        self.c_layout = LayoutEnum.from_tensor(mC) if mC is not None else None
        self.a_major_mode = LayoutEnum.from_tensor(mA).mma_major_mode()
        self.b_major_mode = LayoutEnum.from_tensor(mB).mma_major_mode()

        # Check if input data types are compatible with MMA instruction
        if const_expr(self.a_dtype != self.b_dtype):
            raise TypeError(f"Type must match: {self.a_dtype} != {self.b_dtype}")

        if const_expr(varlen_args is None):
            varlen_args = VarlenArguments()
        assert (varlen_args.mAIdx is not None) == self.gather_A

        # Assume all strides are divisible by 128 bits except the last stride
        new_stride = lambda t: tuple(
            cute.assume(s, divby=128 // t.element_type.width) if not cute.is_static(s) else s
            for s in t.stride
        )
        mA, mD = [
            cute.make_tensor(t.iterator, cute.make_layout(t.shape, stride=new_stride(t)))
            if t is not None
            else None
            for t in (mA, mD)
        ]

        # Setup attributes that dependent on gemm inputs
        self._setup_attributes(epilogue_args, varlen_args)

        if const_expr(self.blockscaled):
            # Setup sfa/sfb tensor by filling A/B tensor to scale factor atom layout
            # ((Atom_M, Rest_M),(Atom_K, Rest_K),RestL)
            sfa_layout = blockscaled_utils.tile_atom_to_shape_SF(mA.shape, self.sf_vec_size)
            mSFA = cute.make_tensor(mSFA.iterator, sfa_layout)
            # ((Atom_N, Rest_N),(Atom_K, Rest_K),RestL)
            sfb_layout = blockscaled_utils.tile_atom_to_shape_SF(mB.shape, self.sf_vec_size)
            mSFB = cute.make_tensor(mSFB.iterator, sfb_layout)

        atom_thr_size = cute.size(self.tiled_mma.thr_id.shape)

        # Setup TMA load for A & B
        a_smem_layout = cute.slice_(self.a_smem_layout_staged, (None, None, None, 0))
        b_smem_layout = cute.slice_(self.b_smem_layout_staged, (None, None, None, 0))
        tma_atom_a, tma_tensor_a = None, None
        if const_expr(not self.gather_A):
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
                internal_type=(cutlass.TFloat32 if mA.element_type is Float32 else None),
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
            internal_type=(cutlass.TFloat32 if mB.element_type is Float32 else None),
        )

        tma_atom_sfa, tma_tensor_sfa = None, None
        tma_atom_sfb, tma_tensor_sfb = None, None
        if const_expr(self.blockscaled):
            # Setup TMA load for SFA
            sfa_op = sm100_utils.cluster_shape_to_tma_atom_A(
                self.cluster_shape_mnk, self.tiled_mma.thr_id
            )
            sfa_smem_layout = cute.slice_(self.sfa_smem_layout_staged, (None, None, None, 0))
            tma_atom_sfa, tma_tensor_sfa = cute.nvgpu.make_tiled_tma_atom_A(
                sfa_op,
                mSFA,
                sfa_smem_layout,
                self.mma_tiler,
                self.tiled_mma,
                self.cluster_layout_vmnk.shape,
                internal_type=cutlass.Int16,
            )
            # Setup TMA load for SFB
            sfb_op = sm100_utils.cluster_shape_to_tma_atom_SFB(
                self.cluster_shape_mnk, self.tiled_mma.thr_id
            )
            sfb_smem_layout = cute.slice_(self.sfb_smem_layout_staged, (None, None, None, 0))
            tma_atom_sfb, tma_tensor_sfb = cute.nvgpu.make_tiled_tma_atom_B(
                sfb_op,
                mSFB,
                sfb_smem_layout,
                self.mma_tiler_sfb,
                self.tiled_mma_sfb,
                self.cluster_layout_sfb_vmnk.shape,
                internal_type=cutlass.Int16,
            )

        self.num_tma_load_bytes = cute.size_in_bytes(self.b_dtype, b_smem_layout)
        if const_expr(not self.gather_A):
            self.num_tma_load_bytes += cute.size_in_bytes(self.a_dtype, a_smem_layout)
        if const_expr(self.blockscaled):
            sfa_copy_size = cute.size_in_bytes(self.sf_dtype, sfa_smem_layout)
            sfb_copy_size = cute.size_in_bytes(self.sf_dtype, sfb_smem_layout)
            self.num_tma_load_bytes += sfa_copy_size + sfb_copy_size
        self.num_tma_load_bytes *= atom_thr_size

        # Setup TMA store for D
        tma_atom_d, tma_tensor_d = None, None
        if const_expr(mD is not None):
            tma_atom_d, tma_tensor_d = self._make_tma_epi_atoms_and_tensors(
                mD,
                self.epi_smem_layout_staged,
                self.epi_tile,
                op_type="store"
                if not (hasattr(epilogue_args, "add_to_output") and epilogue_args.add_to_output)
                else "add",
            )
        tma_atom_c, tma_tensor_c = None, None
        if const_expr(mC is not None):
            tma_atom_c, tma_tensor_c = self._make_tma_epi_atoms_and_tensors(
                mC, self.epi_c_smem_layout_staged, self.epi_tile, op_type="load"
            )

        epilogue_params = self.epi_to_underlying_arguments(epilogue_args)
        varlen_params = VarlenManager.to_underlying_arguments(varlen_args)

        TileSchedulerCls = self.get_scheduler_class(varlen_m=varlen_args.mCuSeqlensM is not None)
        tile_sched_args = self.get_scheduler_arguments(mA, mB, mD, scheduler_args, varlen_args)
        tile_sched_params = TileSchedulerCls.to_underlying_arguments(tile_sched_args)
        grid = TileSchedulerCls.get_grid_shape(
            tile_sched_params, scheduler_args.max_active_clusters
        )

        self.buffer_align_bytes = 1024

        epi_smem_size = cute.cosize(self.epi_smem_layout_staged) if mD is not None else 0
        epi_c_smem_size = cute.cosize(self.epi_c_smem_layout_staged) if mC is not None else 0
        sf_dtype = self.sf_dtype if const_expr(self.blockscaled) else cutlass.Float8E8M0FNU
        sfa_smem_size = (
            cute.cosize(self.sfa_smem_layout_staged) if const_expr(self.blockscaled) else 0
        )
        sfb_smem_size = (
            cute.cosize(self.sfb_smem_layout_staged) if const_expr(self.blockscaled) else 0
        )
        a_idx_smem_size = 0
        if const_expr(self.gather_A):
            a_idx_smem_size = self.a_prefetch_stage * (
                self.cta_tile_shape_mnk[0]
                if varlen_args.mCuSeqlensM is not None
                else self.cta_tile_shape_mnk[2]
            )

        # Define shared storage for kernel
        @cute.struct
        class SharedStorage:
            ab_pipeline_array_ptr: cute.struct.MemRange[cutlass.Int64, self.ab_stage * 2]
            epi_pipeline_array_ptr: cute.struct.MemRange[cutlass.Int64, self.epi_c_stage * 2]
            acc_pipeline_array_ptr: cute.struct.MemRange[cutlass.Int64, self.num_acc_stage * 2]
            sched_pipeline_array_ptr: cute.struct.MemRange[cutlass.Int64, self.sched_stage * 2]
            a_prefetch_pipeline_array_ptr: cute.struct.MemRange[
                cutlass.Int64, self.a_prefetch_stage * 2
            ]
            tile_count: cute.struct.MemRange[Int32, self.sched_stage]
            tmem_dealloc_mbar_ptr: cutlass.Int64
            tmem_holding_buf: Int32
            sAIdx: cute.struct.Align[cute.struct.MemRange[Int32, a_idx_smem_size], 16]
            # (EPI_TILE_M, EPI_TILE_N, STAGE)
            sD: cute.struct.Align[
                cute.struct.MemRange[
                    self.d_dtype if self.d_dtype is not None else Int32, epi_smem_size
                ],
                self.buffer_align_bytes,
            ]
            sC: cute.struct.Align[
                cute.struct.MemRange[
                    self.c_dtype if self.c_dtype is not None else Int32, epi_c_smem_size
                ],
                self.buffer_align_bytes,
            ]
            epi: self.epi_get_smem_struct(epilogue_params)
            # (MMA, MMA_M, MMA_K, STAGE)
            sA: cute.struct.Align[
                cute.struct.MemRange[self.a_dtype, cute.cosize(self.a_smem_layout_staged.outer)],
                self.buffer_align_bytes,
            ]
            # (MMA, MMA_N, MMA_K, STAGE)
            sB: cute.struct.Align[
                cute.struct.MemRange[self.b_dtype, cute.cosize(self.b_smem_layout_staged.outer)],
                self.buffer_align_bytes,
            ]
            # (MMA, MMA_M, MMA_K, STAGE)
            sSFA: cute.struct.Align[
                cute.struct.MemRange[sf_dtype, sfa_smem_size],
                self.buffer_align_bytes,
            ]
            # (MMA, MMA_N, MMA_K, STAGE)
            sSFB: cute.struct.Align[
                cute.struct.MemRange[sf_dtype, sfb_smem_size],
                self.buffer_align_bytes,
            ]

        self.shared_storage = SharedStorage

        # Launch the kernel synchronously
        self.kernel(
            self.tiled_mma,
            self.tiled_mma_sfb,
            tma_atom_a,
            tma_tensor_a if const_expr(not self.gather_A) else mA,
            tma_atom_b,
            tma_tensor_b,
            tma_atom_sfa,
            tma_tensor_sfa,
            tma_atom_sfb,
            tma_tensor_sfb,
            tma_atom_d,
            tma_tensor_d,
            tma_atom_c,
            tma_tensor_c,
            epilogue_params,
            varlen_params,
            self.cluster_layout_vmnk,
            self.cluster_layout_sfb_vmnk,
            self.a_smem_layout_staged,
            self.a_smem_load_layout_staged,
            self.b_smem_layout_staged,
            self.sfa_smem_layout_staged,
            self.sfb_smem_layout_staged,
            self.epi_smem_layout_staged,
            self.epi_c_smem_layout_staged,
            self.epi_tile,
            tile_sched_params,
            TileSchedulerCls,
        ).launch(
            grid=grid,
            block=[self.threads_per_cta, 1, 1],
            cluster=self.cluster_shape_mnk,
            stream=stream,
            min_blocks_per_mp=1,
        )
        return

    # GPU device kernel
    @cute.kernel
    def kernel(
        self,
        tiled_mma: cute.TiledMma,
        tiled_mma_sfb: Optional[cute.TiledMma],
        tma_atom_a: Optional[cute.CopyAtom],
        mA_mkl: cute.Tensor,
        tma_atom_b: cute.CopyAtom,
        mB_nkl: cute.Tensor,
        tma_atom_sfa: Optional[cute.CopyAtom],
        mSFA_mkl: Optional[cute.Tensor],
        tma_atom_sfb: Optional[cute.CopyAtom],
        mSFB_nkl: Optional[cute.Tensor],
        tma_atom_d: Optional[cute.CopyAtom],
        mD_mnl: Optional[cute.Tensor],
        tma_atom_c: Optional[cute.CopyAtom],
        mC_mnl: Optional[cute.Tensor],
        epilogue_params: ParamsBase,
        varlen_params: VarlenManager.Params,
        cluster_layout_vmnk: cute.Layout,
        cluster_layout_sfb_vmnk: Optional[cute.Layout],
        a_smem_layout: cute.ComposedLayout,
        a_smem_load_layout: cute.ComposedLayout,
        b_smem_layout: cute.ComposedLayout,
        sfa_smem_layout: Optional[cute.Layout],
        sfb_smem_layout: Optional[cute.Layout],
        epi_smem_layout: Union[cute.Layout, cute.ComposedLayout, None],
        epi_c_smem_layout: Union[cute.Layout, cute.ComposedLayout, None],
        epi_tile: cute.Tile,
        tile_sched_params: ParamsBase,
        TileSchedulerCls: cutlass.Constexpr[Callable],
    ):
        """
        GPU device kernel performing the Persistent batched GEMM computation.
        """

        varlen_m = const_expr(varlen_params.cu_seqlens_m is not None)
        varlen_k = const_expr(varlen_params.cu_seqlens_k is not None)
        assert not (varlen_m and varlen_k)
        if const_expr(self.gather_A):
            assert varlen_m or varlen_k
        has_D = const_expr(mD_mnl is not None)
        has_C = const_expr(mC_mnl is not None)

        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())

        # /////////////////////////////////////////////////////////////////////////////
        #  Prefetch Tma desc
        # /////////////////////////////////////////////////////////////////////////////
        if warp_idx == self.ab_load_warp_id:
            for tma_atom in (
                tma_atom_a,
                tma_atom_b,
                tma_atom_sfa,
                tma_atom_sfb,
                tma_atom_d,
                tma_atom_c,
            ):
                if const_expr(tma_atom is not None):
                    cpasync.prefetch_descriptor(tma_atom)

        use_2cta_instrs = cute.size(tiled_mma.thr_id.shape) == 2

        #
        # Setup cta/thread coordinates
        #
        # Coords inside cluster
        bidx, _, _ = cute.arch.block_idx()
        mma_tile_coord_v = bidx % cute.size(tiled_mma.thr_id.shape)
        is_leader_cta = mma_tile_coord_v == 0
        cta_rank_in_cluster = cute.arch.make_warp_uniform(cute.arch.block_idx_in_cluster())
        # Coord inside cta
        tidx, _, _ = cute.arch.thread_idx()

        #
        # Alloc and init: a+b full/empty, accumulator full/empty, tensor memory dealloc barrier
        #
        smem = cutlass.utils.SmemAllocator()
        storage = smem.allocate(self.shared_storage)

        tmem_dealloc_mbar_ptr = storage.tmem_dealloc_mbar_ptr
        tmem_holding_buf = storage.tmem_holding_buf

        # Tensor memory dealloc barrier init
        if use_2cta_instrs:
            if warp_idx == self.ab_load_warp_id:
                num_tmem_dealloc_threads = 32
                cute.arch.mbarrier_init(tmem_dealloc_mbar_ptr, num_tmem_dealloc_threads)

        # Initialize pipelines and states
        ab_pipeline = self.make_ab_pipeline(
            tiled_mma=tiled_mma,
            cluster_layout_vmnk=cluster_layout_vmnk,
            ab_pipeline_mbar_ptr=storage.ab_pipeline_array_ptr.data_ptr(),
            is_leader_cta=is_leader_cta,
        )
        epi_pipeline = None
        if const_expr(has_C):
            epi_pipeline = self.make_epi_pipeline(
                c_smem_layout=cute.slice_(epi_c_smem_layout, (None, None, 0)),
                epi_pipeline_mbar_ptr=storage.epi_pipeline_array_ptr.data_ptr(),
            )
        acc_pipeline = self.make_acc_pipeline(
            cluster_layout_vmnk=cluster_layout_vmnk,
            acc_pipeline_mbar_ptr=storage.acc_pipeline_array_ptr.data_ptr(),
        )
        sched_pipeline = None
        tile_count = None
        if const_expr(tile_sched_params.tile_count_semaphore is not None):
            # Dynamic persistent scheduler
            sched_pipeline = self.make_sched_pipeline(
                self.cluster_shape_mnk,
                sched_pipeline_mbar_ptr=storage.sched_pipeline_array_ptr.data_ptr(),
                has_C=has_C,
            )
            tile_count = storage.tile_count.get_tensor((self.sched_stage,))
        a_prefetch_pipeline = None
        if const_expr(self.gather_A):
            a_prefetch_pipeline = self.make_a_prefetch_pipeline(
                storage.a_prefetch_pipeline_array_ptr.data_ptr(),
            )

        # Setup smem tensor A/B/D
        # (MMA, MMA_M, MMA_K, STAGE)
        sA_mma = storage.sA.get_tensor(a_smem_layout.outer, swizzle=a_smem_layout.inner)
        sA = storage.sA.get_tensor(a_smem_load_layout.outer, swizzle=a_smem_load_layout.inner)
        # (MMA, MMA_N, MMA_K, STAGE)
        sB = storage.sB.get_tensor(b_smem_layout.outer, swizzle=b_smem_layout.inner)
        sAIdx = None
        if const_expr(self.gather_A):
            a_idx_smem_dim = self.cta_tile_shape_mnk[0] if varlen_m else self.cta_tile_shape_mnk[2]
            a_idx_smem_layout = cute.make_layout((a_idx_smem_dim, self.a_prefetch_stage))
            sAIdx = storage.sAIdx.get_tensor(a_idx_smem_layout)
        sSFA, sSFB = None, None
        if const_expr(self.blockscaled):
            # (MMA, MMA_M, MMA_K, STAGE)
            sSFA = storage.sSFA.get_tensor(sfa_smem_layout)
            # (MMA, MMA_N, MMA_K, STAGE)
            sSFB = storage.sSFB.get_tensor(sfb_smem_layout)
        sD = None
        if const_expr(has_D):
            # (EPI_TILE_M, EPI_TILE_N, STAGE)
            sD = storage.sD.get_tensor(epi_smem_layout.outer, swizzle=epi_smem_layout.inner)
        sC = None
        if const_expr(has_C):
            sC = storage.sC.get_tensor(epi_c_smem_layout.outer, swizzle=epi_c_smem_layout.inner)
        epi_smem_tensors = self.epi_get_smem_tensors(epilogue_params, storage)

        thr_mma = tiled_mma.get_slice(mma_tile_coord_v)
        thr_mma_sfb = (
            tiled_mma_sfb.get_slice(mma_tile_coord_v) if const_expr(self.blockscaled) else None
        )

        # (MMA, MMA_M, MMA_N)
        acc_shape = tiled_mma.partition_shape_C(self.mma_tiler[:2])
        # (MMA, MMA_M, MMA_N, STAGE)
        tCtAcc_fake = tiled_mma.make_fragment_C(cute.append(acc_shape, self.num_acc_stage))

        varlen_manager = VarlenManager.create(
            varlen_params,
            has_D,
            self.num_epi_tensormaps,
            # Only used if not varlen_m
            len_m_static=Int32(
                mA_mkl.shape[0]
                if varlen_k or varlen_params.mAIdx is None
                else varlen_params.mAIdx.shape[0]
            ),
            len_k_static=Int32(mA_mkl.shape[1]),
        )

        TileSchedulerCls = partial(
            TileSchedulerCls.create, tile_sched_params, tile_count, sched_pipeline
        )

        tmem_alloc_barrier = pipeline.NamedBarrier(
            barrier_id=int(NamedBarrierGemm.TmemPtr),
            num_threads=cute.arch.WARP_SIZE * len((self.mma_warp_id, *self.epilog_warp_id)),
        )
        epi_load_barrier = None
        if const_expr(has_C):
            epi_load_barrier = pipeline.NamedBarrier(
                barrier_id=int(NamedBarrierGemm.EpilogueLoad), num_threads=2 * cute.arch.WARP_SIZE
            )

        #
        # Specialized AB load warps
        #
        if warp_idx == self.ab_load_warp_id:
            is_tma_warp = True
            # initialize tensormap for A & B
            varlen_manager.init_tensormap_AB(tma_atom_a, tma_atom_b, is_tma_warp)
            tma_desc_a_ptr = varlen_manager.get_tma_desc_a_ptr()
            tma_desc_b_ptr = varlen_manager.get_tma_desc_b_ptr()
            # Compute multicast mask for A/B buffer full
            block_in_cluster_coord_vmnk = cluster_layout_vmnk.get_flat_coord(cta_rank_in_cluster)
            block_in_cluster_coord_sfb_vmnk = None
            if const_expr(self.blockscaled):
                block_in_cluster_coord_sfb_vmnk = cluster_layout_sfb_vmnk.get_flat_coord(
                    cta_rank_in_cluster
                )
            a_mcast_mask, b_mcast_mask = None, None
            sfa_mcast_mask, sfb_mcast_mask = None, None
            if const_expr(self.is_a_mcast or self.is_b_mcast or use_2cta_instrs):
                a_mcast_mask = cpasync.create_tma_multicast_mask(
                    cluster_layout_vmnk, block_in_cluster_coord_vmnk, mcast_mode=2
                )
                b_mcast_mask = cpasync.create_tma_multicast_mask(
                    cluster_layout_vmnk, block_in_cluster_coord_vmnk, mcast_mode=1
                )
                if const_expr(self.blockscaled):
                    sfa_mcast_mask = cpasync.create_tma_multicast_mask(
                        cluster_layout_vmnk, block_in_cluster_coord_vmnk, mcast_mode=2
                    )
                    sfb_mcast_mask = cpasync.create_tma_multicast_mask(
                        cluster_layout_sfb_vmnk, block_in_cluster_coord_sfb_vmnk, mcast_mode=1
                    )

            # Persistent tile scheduling loop
            tile_scheduler = TileSchedulerCls()
            work_tile = tile_scheduler.initial_work_tile_info()
            ab_producer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Producer, self.ab_stage
            )
            if const_expr(varlen_k):
                # wait tensormap initialization complete before update
                varlen_manager.fence_tensormap_init()
            do_epi_load_barrier_arrive = Boolean(True)
            while work_tile.is_valid_tile:
                tile_coord_mnkl = work_tile.tile_idx
                batch_idx = tile_coord_mnkl[3]
                varlen_manager.update_tensormap_AB(
                    batch_idx,
                    self.a_layout,
                    self.b_layout,
                    is_tma_warp,
                )
                # ///////////////////////////////////////////////////////////////////////////
                #  Local_tile partition global tensors
                # ///////////////////////////////////////////////////////////////////////////
                mma_tile_coord_mnl = (
                    tile_coord_mnkl[0] // cute.size(tiled_mma.thr_id.shape),
                    tile_coord_mnkl[1],
                    tile_coord_mnkl[3],
                )
                gA_mk = None
                if const_expr(not self.gather_A):
                    mA_mk = varlen_manager.offset_batch_A(mA_mkl, batch_idx)
                    # (bM, bK, RestK)
                    gA_mk = cute.local_tile(
                        mA_mk,
                        cute.select(self.mma_tiler, [0, 2]),
                        (mma_tile_coord_mnl[0], None),
                    )
                # (bN, bK, RestK)
                gB_nk = cute.local_tile(
                    varlen_manager.offset_batch_B(mB_nkl, batch_idx),
                    cute.select(self.mma_tiler, [1, 2]),
                    (mma_tile_coord_mnl[1], None),
                )
                if const_expr(self.blockscaled):
                    # (bM, bK)
                    gSFA_mkl = cute.local_tile(
                        varlen_manager.offset_batch_A(mSFA_mkl, batch_idx),
                        cute.select(self.mma_tiler, [0, 2]),
                        (mma_tile_coord_mnl[0], None),
                    )
                    # (bN, bK)
                    gSFB_nkl = cute.local_tile(
                        varlen_manager.offset_batch_B(mSFB_nkl, batch_idx),
                        cute.select(self.mma_tiler, [1, 2]),
                        (mma_tile_coord_mnl[1], None),
                    )

                # Partition global tensor for TiledMMA_A/B/D
                # Then partition global/shared tensor for TMA load A/B
                varlen_manager.fence_tensormap_update_AB(is_tma_warp)
                len_k = varlen_manager.len_k(batch_idx)
                # TMA load A partition_S/D
                a_cta_layout = cute.make_layout(
                    cute.slice_(cluster_layout_vmnk, (0, 0, None, 0)).shape
                )
                copy_A = None
                if const_expr(not self.gather_A):
                    # (MMA, MMA_M, MMA_K, RestK)
                    tCgA = thr_mma.partition_A(gA_mk)
                    copy_A, _, _ = copy_utils_tma_get_copy_fn(
                        tma_atom_a,
                        cta_coord=block_in_cluster_coord_vmnk[2],
                        cta_layout=a_cta_layout,
                        src_tensor=tCgA,
                        dst_tensor=sA,
                        mcast_mask=a_mcast_mask,
                        tma_desc_ptr=tma_desc_a_ptr,
                    )
                # (MMA, MMA_N, MMA_K, RestK)
                tCgB = thr_mma.partition_B(gB_nk)
                if const_expr(self.blockscaled):
                    # (MMA, MMA_M, MMA_K)
                    tCgSFA = thr_mma.partition_A(gSFA_mkl)
                    # (MMA, MMA_N, MMA_K)
                    tCgSFB = thr_mma_sfb.partition_B(gSFB_nkl)
                # TMA load B partition_S/D
                copy_B, _, _ = copy_utils_tma_get_copy_fn(
                    tma_atom_b,
                    cta_coord=block_in_cluster_coord_vmnk[1],
                    cta_layout=cute.make_layout(
                        cute.slice_(cluster_layout_vmnk, (0, None, 0, 0)).shape
                    ),
                    src_tensor=tCgB,
                    dst_tensor=sB,
                    mcast_mask=b_mcast_mask,
                    tma_desc_ptr=tma_desc_b_ptr,
                )
                copy_SFA, copy_SFB = None, None
                if const_expr(self.blockscaled):
                    #  TMA load SFA partition_S/D
                    copy_SFA, _, _ = copy_utils.tma_get_copy_fn(
                        tma_atom_sfa,
                        cta_coord=block_in_cluster_coord_vmnk[2],
                        cta_layout=a_cta_layout,
                        src_tensor=tCgSFA,
                        dst_tensor=sSFA,
                        filter_zeros=True,
                        mcast_mask=sfa_mcast_mask,
                        # tma_desc_ptr=tma_desc_sfa_ptr,
                    )
                    # TMA load SFB partition_S/D
                    sfb_cta_layout = cute.make_layout(
                        cute.slice_(cluster_layout_sfb_vmnk, (0, None, 0, 0)).shape
                    )
                    copy_SFB, _, _ = copy_utils.tma_get_copy_fn(
                        tma_atom_sfb,
                        cta_coord=block_in_cluster_coord_sfb_vmnk[1],
                        cta_layout=sfb_cta_layout,
                        src_tensor=tCgSFB,
                        dst_tensor=sSFB,
                        filter_zeros=True,
                        mcast_mask=sfb_mcast_mask,
                        # tma_desc_ptr=tma_desc_sfa_ptr,
                    )
                k_tile_cnt = cute.ceil_div(len_k, self.cta_tile_shape_mnk[2])
                ab_producer_state = self.load_AB(
                    ab_pipeline,
                    ab_producer_state,
                    copy_A,
                    copy_B,
                    k_tile_cnt,
                    copy_SFA,
                    copy_SFB,
                )
                if const_expr(epi_load_barrier is not None):
                    # In the first work tile, the epi load warp will wait for the signal
                    # from the mainloop load warp to start loading C, to avoid interfering
                    # with loading A and B.
                    if do_epi_load_barrier_arrive:
                        epi_load_barrier.arrive()
                        do_epi_load_barrier_arrive = Boolean(False)
                # Advance to next tile
                tile_scheduler.advance_to_next_work()
                work_tile = tile_scheduler.get_current_work()
            # Wait A/B buffer empty
            ab_pipeline.producer_tail(ab_producer_state)

        if const_expr(self.gather_A):
            if (
                warp_idx >= self.ab_load_warp_id + 1
                and warp_idx < self.ab_load_warp_id + self.num_ab_load_warps
            ):
                # Persistent tile scheduling loop
                tile_scheduler = TileSchedulerCls()
                work_tile = tile_scheduler.initial_work_tile_info()
                ab_producer_state = pipeline.make_pipeline_state(
                    pipeline.PipelineUserType.Producer, self.ab_stage
                )
                a_prefetch_consumer_state = pipeline.make_pipeline_state(
                    pipeline.PipelineUserType.Consumer, self.a_prefetch_stage
                )
                while work_tile.is_valid_tile:
                    tile_coord_mnkl = work_tile.tile_idx
                    batch_idx = tile_coord_mnkl[3]
                    # ///////////////////////////////////////////////////////////////////////////
                    #  Local_tile partition global tensors
                    # ///////////////////////////////////////////////////////////////////////////
                    mAIdx_mk = varlen_manager.offset_batch_AIdx(batch_idx)
                    if const_expr(varlen_m):
                        # (M, K)
                        mA_mk = mA_mkl
                    else:
                        assert varlen_k
                        # (tile_M, K)
                        mA_mk = cute.local_tile(
                            mA_mkl, (self.cta_tile_shape_mnk[0],), (tile_coord_mnkl[0], None)
                        )
                    # Partition global tensor for TiledMMA_A/B/D
                    len_m = varlen_manager.len_m(batch_idx)
                    len_k = varlen_manager.len_k(batch_idx)
                    # TMA load A partition_S/D
                    tiled_copy_A = self._make_gmem_tiled_copy_A(
                        mA_mkl.element_type, self.a_layout, (self.num_ab_load_warps - 1) * 32
                    )
                    tidx = cute.arch.thread_idx()[0] - (self.ab_load_warp_id + 1) * 32
                    thr_copy_A = tiled_copy_A.get_slice(tidx)
                    copy_A, prefetch_A = None, None
                    if const_expr(varlen_m):
                        a_prefetch_pipeline.consumer_wait(a_prefetch_consumer_state)
                        copy_A = copy_utils.gather_m_get_copy_fn(
                            thr_copy_A,
                            mA_mk,
                            sA,
                            sAIdx[None, a_prefetch_consumer_state.index],
                            limit_m=len_m - tile_coord_mnkl[0] * self.cta_tile_shape_mnk[0],
                            limit_k=len_k,
                        )
                        cute.arch.sync_warp()
                        with cute.arch.elect_one():
                            a_prefetch_pipeline.consumer_release(a_prefetch_consumer_state)
                        a_prefetch_consumer_state.advance()
                    else:
                        copy_A, prefetch_A = copy_utils.gather_k_get_copy_fn(
                            thr_copy_A,
                            mA_mk,
                            sA,
                            sAIdx,
                            limit_m=len_m - tile_coord_mnkl[0] * self.cta_tile_shape_mnk[0],
                            limit_k=len_k,
                        )
                        prefetch_A = partial(prefetch_A, a_prefetch_pipeline)
                    k_tile_cnt = cute.ceil_div(len_k, self.cta_tile_shape_mnk[2])
                    ab_producer_state, a_prefetch_consumer_state = self.load_A_gather_A(
                        ab_pipeline,
                        ab_producer_state,
                        a_prefetch_consumer_state,
                        copy_A,
                        prefetch_A,
                        k_tile_cnt,
                    )
                    # Advance to next tile
                    tile_scheduler.advance_to_next_work()
                    work_tile = tile_scheduler.get_current_work()

        #
        # Specialized scheduler warp. Will also prefetch A indices if gatherA
        #
        if const_expr(tile_sched_params.tile_count_semaphore is not None or self.gather_A):
            if warp_idx == self.scheduler_warp_id:
                is_scheduler_warp = True
                if const_expr(cute.size(cluster_layout_vmnk) > 1):
                    is_scheduler_warp = cute.arch.block_idx_in_cluster() == 0
                tile_M = self.cta_tile_shape_mnk[0]
                tile_K = self.cta_tile_shape_mnk[2]
                thr_copy_AIdx, tAsAIdx, tAcAIdx = None, None, None
                if const_expr(self.gather_A):
                    tiled_copy_AIdx = copy_utils.tiled_copy_1d(Int32, num_threads=32, is_async=True)
                    thr_copy_AIdx = tiled_copy_AIdx.get_slice(cute.arch.lane_idx())
                    tAsAIdx = thr_copy_AIdx.partition_D(sAIdx)
                    tAcAIdx = thr_copy_AIdx.partition_S(
                        cute.make_identity_tensor(tile_M if varlen_m else tile_K)
                    )
                # Persistent tile scheduling loop
                tile_scheduler = TileSchedulerCls(is_scheduler_warp=is_scheduler_warp)
                work_tile = tile_scheduler.initial_work_tile_info()
                a_prefetch_producer_state = None
                if const_expr(self.gather_A):
                    a_prefetch_producer_state = pipeline.make_pipeline_state(
                        pipeline.PipelineUserType.Producer, self.a_prefetch_stage
                    )
                while work_tile.is_valid_tile:
                    if const_expr(self.gather_A):
                        tile_coord_mnkl = work_tile.tile_idx
                        batch_idx = tile_coord_mnkl[3]
                        mAIdx_mk = varlen_manager.offset_batch_AIdx(batch_idx)
                        if const_expr(varlen_m):
                            # (tile_M,)
                            gAIdx = cute.local_tile(mAIdx_mk, (tile_M,), (tile_coord_mnkl[0],))
                            tAgAIdx = thr_copy_AIdx.partition_S(gAIdx)
                            len_m = varlen_manager.len_m(batch_idx)
                            m_limit = len_m - tile_coord_mnkl[0] * tile_M
                            tApAIdx_m = cute.make_fragment((1, tAsAIdx.shape[1]), Boolean)
                            for m in cutlass.range(tAsAIdx.shape[1], unroll_full=True):
                                tApAIdx_m[0, m] = tAcAIdx[0, m] < m_limit
                            a_prefetch_pipeline.producer_acquire(a_prefetch_producer_state)
                            cute.copy(
                                thr_copy_AIdx,
                                tAgAIdx,
                                tAsAIdx[None, None, a_prefetch_producer_state.index],
                                pred=tApAIdx_m,
                            )
                            a_prefetch_pipeline.producer_commit(a_prefetch_producer_state)
                            a_prefetch_producer_state.advance()
                        else:
                            # (tile_K, RestK)
                            gAIdx = cute.flat_divide(mAIdx_mk, (tile_K,))
                            tAgAIdx = thr_copy_AIdx.partition_S(gAIdx)
                            len_k = varlen_manager.len_k(batch_idx)
                            k_tile_cnt = cute.ceil_div(len_k, tile_K)
                            for k_tile in cutlass.range(k_tile_cnt - 1, unroll=1):
                                a_prefetch_pipeline.producer_acquire(a_prefetch_producer_state)
                                cute.copy(
                                    thr_copy_AIdx,
                                    tAgAIdx[None, None, k_tile],
                                    tAsAIdx[None, None, a_prefetch_producer_state.index],
                                )
                                a_prefetch_pipeline.producer_commit(a_prefetch_producer_state)
                                a_prefetch_producer_state.advance()
                            if 0 < k_tile_cnt:
                                k_tile = k_tile_cnt - 1
                                k_limit = len_k - k_tile * tile_K
                                tApAIdx_k = cute.make_fragment((1, tAsAIdx.shape[1]), Boolean)
                                for m in cutlass.range(tAsAIdx.shape[1], unroll_full=True):
                                    tApAIdx_k[0, m] = tAcAIdx[0, m] < k_limit
                                a_prefetch_pipeline.producer_acquire(a_prefetch_producer_state)
                                cute.copy(
                                    tiled_copy_AIdx,
                                    tAgAIdx[None, None, k_tile],
                                    tAsAIdx[None, None, a_prefetch_producer_state.index],
                                    pred=tApAIdx_k,
                                )
                                a_prefetch_pipeline.producer_commit(a_prefetch_producer_state)
                                a_prefetch_producer_state.advance()
                    # Advance to next tile
                    tile_scheduler.fetch_next_work(is_scheduler_warp=is_scheduler_warp)
                    tile_scheduler.advance_to_next_work(is_scheduler_warp=is_scheduler_warp)
                    work_tile = tile_scheduler.get_current_work()
                    # End of persistent scheduler loop
                if is_scheduler_warp:
                    tile_scheduler.producer_tail()

        #
        # Specialized TMA epi load warp
        #
        if const_expr(mC_mnl is not None):
            if warp_idx == self.epi_load_warp_id:
                epi_producer_state = pipeline.make_pipeline_state(
                    pipeline.PipelineUserType.Producer, self.epi_c_stage
                )
                do_epi_load_barrier_wait = Boolean(True)
                # Persistent tile scheduling loop
                tile_scheduler = TileSchedulerCls()
                work_tile = tile_scheduler.initial_work_tile_info()
                while work_tile.is_valid_tile:
                    # Get tile coord from tile scheduler
                    tile_coord_mnkl = work_tile.tile_idx
                    batch_idx = tile_coord_mnkl[3]
                    copy_C_fn, _, bGS_gC = self.epilog_gmem_copy_and_partition(
                        tma_atom_c,
                        varlen_manager.offset_batch_epi(mC_mnl, batch_idx),
                        self.cta_tile_shape_mnk[:2],
                        epi_tile,
                        sC,
                        tile_coord_mnkl,
                    )
                    copy_C = copy_utils.tma_producer_copy_fn(copy_C_fn, epi_pipeline)
                    if do_epi_load_barrier_wait:
                        epi_load_barrier.arrive_and_wait()
                        do_epi_load_barrier_wait = Boolean(False)
                    epi_tile_num = const_expr(cute.size(bGS_gC, mode=[1]))
                    for epi_idx in cutlass.range(epi_tile_num, unroll=1):
                        epi_pipeline.producer_acquire(epi_producer_state)
                        copy_C(src_idx=epi_idx, producer_state=epi_producer_state)
                        # Epi pipeline's producer commit is a NOP
                        epi_pipeline.producer_commit(epi_producer_state)
                        epi_producer_state.advance()
                    # Advance to next tile
                    tile_scheduler.advance_to_next_work()
                    work_tile = tile_scheduler.get_current_work()
                    # End of persistent scheduler loop
                epi_pipeline.producer_tail(epi_producer_state)

        #
        # Specialized MMA warp
        #
        if warp_idx == self.mma_warp_id:
            tmem_alloc_barrier.arrive_and_wait()
            # Retrieving tensor memory ptr and make accumulator tensor
            acc_tmem_ptr = cute.arch.retrieve_tmem_ptr(
                self.acc_dtype, alignment=16, ptr_to_buffer_holding_addr=tmem_holding_buf
            )
            # Partition shared/tensor memory tensor for TiledMMA_A/B/D
            # (MMA, MMA_M, MMA_K, STAGE)
            tCrA = tiled_mma.make_fragment_A(sA_mma)
            # (MMA, MMA_N, MMA_K, STAGE)
            tCrB = tiled_mma.make_fragment_B(sB)
            # (MMA, MMA_M, MMA_N, STAGE)
            tCtAcc_base = cute.make_tensor(acc_tmem_ptr, tCtAcc_fake.layout)

            if const_expr(self.blockscaled):
                # Make SFA tmem tensor
                sfa_tmem_ptr = cute.recast_ptr(
                    acc_tmem_ptr + tcgen05.find_tmem_tensor_col_offset(tCtAcc_base),
                    dtype=self.sf_dtype,
                )
                # (MMA, MMA_M, MMA_K)
                tCtSFA_layout = blockscaled_utils.make_tmem_layout_sfa(
                    tiled_mma,
                    self.mma_tiler,
                    self.sf_vec_size,
                    cute.slice_(sfa_smem_layout, (None, None, None, 0)),
                )
                tCtSFA = cute.make_tensor(sfa_tmem_ptr, tCtSFA_layout)
                # Make SFB tmem tensor
                sfb_tmem_ptr = cute.recast_ptr(
                    acc_tmem_ptr
                    + tcgen05.find_tmem_tensor_col_offset(tCtAcc_base)
                    + tcgen05.find_tmem_tensor_col_offset(tCtSFA),
                    dtype=self.sf_dtype,
                )
                # (MMA, MMA_N, MMA_K)
                tCtSFB_layout = blockscaled_utils.make_tmem_layout_sfb(
                    tiled_mma,
                    self.mma_tiler,
                    self.sf_vec_size,
                    cute.slice_(sfb_smem_layout, (None, None, None, 0)),
                )
                tCtSFB = cute.make_tensor(sfb_tmem_ptr, tCtSFB_layout)
                # Partition for S2T copy of SFA/SFB
                (
                    tiled_copy_s2t_sfa,
                    tCsSFA_compact_s2t,
                    tCtSFA_compact_s2t,
                ) = self.mainloop_s2t_copy_and_partition(sSFA, tCtSFA)
                (
                    tiled_copy_s2t_sfb,
                    tCsSFB_compact_s2t,
                    tCtSFB_compact_s2t,
                ) = self.mainloop_s2t_copy_and_partition(sSFB, tCtSFB)
            else:
                tCtSFA, tCtSFB = None, None
                tiled_copy_s2t_sfa, tCsSFA_compact_s2t, tCtSFA_compact_s2t = None, None, None
                tiled_copy_s2t_sfb, tCsSFB_compact_s2t, tCtSFB_compact_s2t = None, None, None

            # Persistent tile scheduling loop
            tile_scheduler = TileSchedulerCls()
            work_tile = tile_scheduler.initial_work_tile_info()
            ab_consumer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Consumer, self.ab_stage
            )
            acc_producer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Producer, self.num_acc_stage
            )
            while work_tile.is_valid_tile:
                # Get tile coord from tile scheduler
                tile_coord_mnkl = work_tile.tile_idx
                batch_idx = tile_coord_mnkl[3]
                k_len = varlen_manager.len_k(batch_idx)
                k_tile_cnt = cute.ceil_div(k_len, self.mma_tiler[2])
                # Set tensor memory buffer for current tile
                # (MMA, MMA_M, MMA_N)
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
                    is_leader_cta,
                    cta_rank_in_cluster,
                    tCtSFA,
                    tCtSFB,
                    tiled_copy_s2t_sfa,
                    tiled_copy_s2t_sfb,
                    tCsSFA_compact_s2t,
                    tCsSFB_compact_s2t,
                    tCtSFA_compact_s2t,
                    tCtSFB_compact_s2t,
                )
                # Advance to next tile
                tile_scheduler.advance_to_next_work()
                work_tile = tile_scheduler.get_current_work()

            # Wait for accumulator buffer empty
            acc_pipeline.producer_tail(acc_producer_state)

        #
        # Specialized epilogue warps
        #
        if warp_idx < self.mma_warp_id:
            # Alloc tensor memory buffer
            if warp_idx == self.epilog_warp_id[0]:
                cute.arch.alloc_tmem(
                    self.num_tmem_alloc_cols, tmem_holding_buf, is_two_cta=use_2cta_instrs
                )
            # Bar sync for retrieve tensor memory ptr from shared memory
            tmem_alloc_barrier.arrive_and_wait()

            is_tma_warp = Boolean(warp_idx == self.epilog_warp_id[0])
            varlen_manager.init_tensormap_epi(
                tma_atom_d, self.epi_get_tma_atoms(epilogue_params), is_tma_warp
            )
            tma_desc_d_ptr = varlen_manager.get_tma_desc_d_ptr()
            tma_desc_epi_ptrs = varlen_manager.get_tma_desc_epi_ptrs()

            # Retrieving tensor memory ptr and make accumulator tensor
            acc_tmem_ptr = cute.arch.retrieve_tmem_ptr(
                self.acc_dtype, alignment=16, ptr_to_buffer_holding_addr=tmem_holding_buf
            )
            # (MMA, MMA_M, MMA_N, STAGE)
            tCtAcc_base = cute.make_tensor(acc_tmem_ptr, tCtAcc_fake.layout)

            epilogue_barrier = pipeline.NamedBarrier(
                barrier_id=int(NamedBarrierGemm.Epilogue),
                num_threads=self.num_epi_warps * cute.arch.WARP_SIZE,
            )

            # Partition for epilogue
            epi_tidx = tidx
            tiled_copy_t2r, tTR_tAcc_base, tTR_rAcc = self.epilog_tmem_copy_and_partition(
                epi_tidx, tCtAcc_base, epi_tile, use_2cta_instrs
            )

            tTR_rD = cute.make_fragment(tTR_rAcc.shape, self.acc_dtype)
            tiled_copy_r2s, tRS_rD, tRS_sD = self.epilog_smem_store_and_partition(
                tiled_copy_t2r, self.d_layout, self.d_dtype, tTR_rD, sD, epi_tidx
            )
            tRS_rC, tSR_rC, tSR_sC = None, None, None
            tiled_copy_s2r = None
            if const_expr(mC_mnl is not None):
                tiled_copy_s2r, tRS_rC, tSR_rC, tSR_sC = self.epilog_smem_load_and_partition(
                    tiled_copy_t2r, self.c_layout, self.c_dtype, sC, tRS_rD.layout, epi_tidx
                )

            # Persistent tile scheduling loop
            tile_scheduler = TileSchedulerCls()
            work_tile = tile_scheduler.initial_work_tile_info()
            acc_consumer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Consumer, self.num_acc_stage
            )
            epi_store_pipeline = self.make_epi_store_pipeline()
            epi_read_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Consumer, self.epi_c_stage
            )
            if const_expr(varlen_m):
                # wait tensormap initialization complete before update
                varlen_manager.fence_tensormap_init()
            while work_tile.is_valid_tile:
                # Get tile coord from tile scheduler
                tile_coord_mnkl = work_tile.tile_idx
                batch_idx = tile_coord_mnkl[3]
                epi_shapes, epi_orders = self.epi_get_tensormap_update_shapes_orders(
                    epilogue_params, varlen_params.cu_seqlens_m, batch_idx
                )
                varlen_manager.update_tensormap_epi(
                    batch_idx,
                    self.d_layout,
                    epi_shapes,
                    epi_orders,
                    is_tma_warp,
                )

                # Set tensor memory buffer for current tile
                # (T2R, T2R_M, T2R_N, EPI_M, EPI_M)
                tTR_tAcc = tTR_tAcc_base[None, None, None, None, None, acc_consumer_state.index]

                # Wait for accumulator buffer full
                acc_pipeline.consumer_wait(acc_consumer_state)

                varlen_manager.fence_tensormap_update_epi(is_tma_warp)

                copy_D = None
                if const_expr(has_D):
                    copy_D, _, _ = self.epilog_gmem_copy_and_partition(
                        tma_atom_d,
                        varlen_manager.offset_batch_epi(mD_mnl, batch_idx),
                        self.cta_tile_shape_mnk[:2],
                        epi_tile,
                        sD,
                        tile_coord_mnkl,
                        tma_desc_ptr=tma_desc_d_ptr,
                    )
                copy_C = None  # We're using a separate warp to load C

                tTR_tAcc = cute.group_modes(tTR_tAcc, 3, cute.rank(tTR_tAcc))
                k_len = varlen_manager.len_k(batch_idx)
                load_acc_subtile = partial(
                    self.epi_load_acc_subtile,
                    tiled_copy_t2r,
                    tiled_copy_r2s,
                    tTR_tAcc,
                    tTR_rAcc,
                    clear_acc=varlen_k and k_len == 0,
                )

                epi_read_state, _ = self.epilogue(
                    epilogue_params,
                    epi_smem_tensors,
                    tma_desc_epi_ptrs,
                    epi_pipeline,
                    epi_store_pipeline,
                    epi_read_state,
                    None,  # epi_producer_state
                    epi_tile,
                    load_acc_subtile,
                    tRS_rD,
                    tRS_rC,
                    tiled_copy_t2r,
                    tiled_copy_r2s,
                    tRS_sD,
                    tiled_copy_s2r,
                    tSR_rC,
                    tSR_sC,
                    copy_D,
                    copy_C,
                    tile_coord_mnkl,
                    varlen_manager,
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
                cute.arch.relinquish_tmem_alloc_permit(is_two_cta=use_2cta_instrs)
            epilogue_barrier.arrive_and_wait()
            if warp_idx == self.epilog_warp_id[0]:
                if const_expr(use_2cta_instrs):
                    cute.arch.mbarrier_arrive(tmem_dealloc_mbar_ptr, cta_rank_in_cluster ^ 1)
                    cute.arch.mbarrier_wait(tmem_dealloc_mbar_ptr, 0)
                cute.arch.dealloc_tmem(
                    acc_tmem_ptr, self.num_tmem_alloc_cols, is_two_cta=use_2cta_instrs
                )

            # Wait for D store complete
            if is_tma_warp:
                epi_store_pipeline.producer_tail()

    @cute.jit
    def load_A_gather_A(
        self,
        a_pipeline: cutlass.pipeline.PipelineAsync,
        a_producer_state: cutlass.pipeline.PipelineState,
        a_prefetch_consumer_state: Optional[cutlass.pipeline.PipelineState],
        copy_A: Callable,
        prefetch_A: Optional[Callable],
        k_tile_cnt: Int32,
    ) -> Tuple[cutlass.pipeline.PipelineState, Optional[cutlass.pipeline.PipelineState]]:
        # Peek (try_wait) AB buffer empty for k_block = prefetch_k_tile_cnt
        peek_a_empty_status = Boolean(True)
        if 0 < k_tile_cnt:
            peek_a_empty_status = a_pipeline.producer_try_acquire(a_producer_state)
        # /////////////////////////////////////////////////////////////////////////
        # cp.async on A
        # /////////////////////////////////////////////////////////////////////////
        is_tma_warp = False
        for k_tile in cutlass.range(k_tile_cnt - 1, unroll=1):
            smem_idx = a_producer_state.index
            prefetch_out = ()
            if const_expr(prefetch_A is not None):  # Prefetch early, even before smem is free
                prefetch_out = (prefetch_A(k_tile, smem_idx, a_prefetch_consumer_state),)
                a_prefetch_consumer_state.advance()
            a_pipeline.producer_acquire(a_producer_state, peek_a_empty_status, is_tma_warp)
            copy_A(k_tile, smem_idx, *prefetch_out)
            # This tells mbarrier to track the completion of cp.async
            a_pipeline.producer_cpasync_commit(a_producer_state)
            a_producer_state.advance()
            peek_a_empty_status = Boolean(True)
            if k_tile + 1 < k_tile_cnt:
                peek_a_empty_status = a_pipeline.producer_try_acquire(a_producer_state)
        # bound checking in the K dimension on the last k_tile
        if 0 < k_tile_cnt:
            k_tile = k_tile_cnt - 1
            smem_idx = a_producer_state.index
            prefetch_out = ()
            if const_expr(prefetch_A is not None):  # Prefetch early, even before smem is free
                prefetch_out = (prefetch_A(k_tile, smem_idx, a_prefetch_consumer_state, pred=True),)
                a_prefetch_consumer_state.advance()
            a_pipeline.producer_acquire(a_producer_state, peek_a_empty_status, is_tma_warp)
            copy_A(k_tile, smem_idx, *prefetch_out, pred=True)
            a_pipeline.producer_cpasync_commit(a_producer_state)
            a_producer_state.advance()
        return a_producer_state, a_prefetch_consumer_state

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
        is_leader_cta: Boolean,
        cta_rank_in_cluster: Int32,
        tCtSFA: Optional[cute.Tensor] = None,
        tCtSFB: Optional[cute.Tensor] = None,
        tiled_copy_s2t_sfa: Optional[cute.TiledCopy] = None,
        tiled_copy_s2t_sfb: Optional[cute.TiledCopy] = None,
        tCsSFA_compact_s2t: Optional[cute.Tensor] = None,
        tCsSFB_compact_s2t: Optional[cute.Tensor] = None,
        tCtSFA_compact_s2t: Optional[cute.Tensor] = None,
        tCtSFB_compact_s2t: Optional[cute.Tensor] = None,
    ) -> Tuple[cutlass.pipeline.PipelineState, cutlass.pipeline.PipelineState, cute.TiledMma]:
        blockscaled = const_expr(tiled_copy_s2t_sfa is not None)
        if const_expr(blockscaled):
            assert all(x is not None for x in (tCtSFA, tCtSFB))
            assert all(x is not None for x in (tiled_copy_s2t_sfa, tiled_copy_s2t_sfb))
            assert all(x is not None for x in (tCsSFA_compact_s2t, tCsSFB_compact_s2t))
            assert all(x is not None for x in (tCtSFA_compact_s2t, tCtSFB_compact_s2t))
        # If gather_A and use_2cta_instrs, the cp.async for the non-leader CTA will
        # arrive at an mbarrier on the non-leader CTA side, then the mma warp of the non-leader
        # CTA will wait for that then arrive at the mbarrier on the leader CTA.
        need_nonleader_cta = const_expr(self.gather_A and self.use_2cta_instrs)
        # Peek (try_wait) AB buffer full for k_tile = 0
        peek_ab_full_status = Boolean(True)
        if 0 < k_tile_cnt and (is_leader_cta or need_nonleader_cta):
            peek_ab_full_status = ab_pipeline.consumer_try_wait(ab_consumer_state)
        # Wait for accumulator buffer empty
        if is_leader_cta:
            acc_pipeline.producer_acquire(acc_producer_state)
        # Reset the ACCUMULATE field for each tile
        tiled_mma.set(tcgen05.Field.ACCUMULATE, False)
        # Mma mainloop
        num_k_blocks = cute.size(tCrA, mode=[2])
        for k_tile in cutlass.range(k_tile_cnt, unroll=1):
            if const_expr(need_nonleader_cta):
                if not is_leader_cta:
                    ab_pipeline.consumer_wait(ab_consumer_state, peek_ab_full_status)
                    with cute.arch.elect_one():
                        # The odd CTA signals the even CTA
                        ab_pipeline.sync_object_full.arrive_mbarrier(
                            ab_consumer_state.index, dst_rank=cta_rank_in_cluster & 0xFE
                        )
            if is_leader_cta:
                # Conditionally wait for AB buffer full
                ab_pipeline.consumer_wait(ab_consumer_state, peek_ab_full_status)
                #  Copy SFA/SFB from smem to tmem
                if const_expr(blockscaled):
                    s2t_stage_coord = (None, None, None, None, ab_consumer_state.index)
                    tCsSFA_compact_s2t_staged = tCsSFA_compact_s2t[s2t_stage_coord]
                    tCsSFB_compact_s2t_staged = tCsSFB_compact_s2t[s2t_stage_coord]
                    cute.copy(tiled_copy_s2t_sfa, tCsSFA_compact_s2t_staged, tCtSFA_compact_s2t)
                    cute.copy(tiled_copy_s2t_sfb, tCsSFB_compact_s2t_staged, tCtSFB_compact_s2t)
                for k_blk_idx in cutlass.range(num_k_blocks, unroll_full=True):
                    k_blk_coord = (None, None, k_blk_idx, ab_consumer_state.index)
                    if const_expr(blockscaled):
                        # Set SFA/SFB tensor to tiled_mma
                        sf_kblock_coord = (None, None, k_blk_idx)
                        tiled_mma.set(tcgen05.Field.SFA, tCtSFA[sf_kblock_coord].iterator)
                        tiled_mma.set(tcgen05.Field.SFB, tCtSFB[sf_kblock_coord].iterator)
                    cute.gemm(tiled_mma, acc, tCrA[k_blk_coord], tCrB[k_blk_coord], acc)
                    tiled_mma.set(tcgen05.Field.ACCUMULATE, True)
                # Async arrive AB buffer empty
                ab_pipeline.consumer_release(ab_consumer_state)
            ab_consumer_state.advance()
            # Peek (try_wait) AB buffer full for k_tile = k_tile + 1
            peek_ab_full_status = Boolean(True)
            if k_tile + 1 < k_tile_cnt and (is_leader_cta or need_nonleader_cta):
                peek_ab_full_status = ab_pipeline.consumer_try_wait(ab_consumer_state)
        # Async arrive accumulator buffer full
        if is_leader_cta:
            acc_pipeline.producer_commit(acc_producer_state)
        acc_producer_state.advance()
        # If we don't return the tiled_mma, we get compiler error
        # "operand #0 does not dominate this use"
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
        clear_acc: Boolean = False,
    ):
        if not clear_acc:
            # Load accumulator from tensor memory buffer to register
            cute.copy(tiled_copy_t2r, tTR_tAcc[None, None, None, epi_idx], tTR_rAcc)
            tRS_rAcc = tiled_copy_r2s.retile(tTR_rAcc)
            tRS_rD.store(tRS_rAcc.load())
        else:
            tRS_rD.fill(0.0)

    def mainloop_s2t_copy_and_partition(
        self,
        sSF: cute.Tensor,
        tSF: cute.Tensor,
    ) -> Tuple[cute.TiledCopy, cute.Tensor, cute.Tensor]:
        """
        Make tiledCopy for smem to tmem load for scale factor tensor, then use it to partition smem memory (source) and tensor memory (destination).

        :param sSF: The scale factor tensor in smem
        :type sSF: cute.Tensor
        :param tSF: The scale factor tensor in tmem
        :type tSF: cute.Tensor

        :return: A tuple containing (tiled_copy_s2t, tCsSF_compact_s2t, tCtSF_compact_s2t) where:
            - tiled_copy_s2t: The tiled copy operation for smem to tmem load for scale factor tensor(s2t)
            - tCsSF_compact_s2t: The partitioned scale factor tensor in smem
            - tSF_compact_s2t: The partitioned scale factor tensor in tmem
        :rtype: Tuple[cute.TiledCopy, cute.Tensor, cute.Tensor]
        """
        # (MMA, MMA_MN, MMA_K, STAGE)
        tCsSF_compact = cute.filter_zeros(sSF)
        # (MMA, MMA_MN, MMA_K)
        tCtSF_compact = cute.filter_zeros(tSF)
        # Make S2T CopyAtom and tiledCopy
        copy_atom_s2t = cute.make_copy_atom(tcgen05.Cp4x32x128bOp(self.cta_group), self.sf_dtype)
        tiled_copy_s2t = tcgen05.make_s2t_copy(copy_atom_s2t, tCtSF_compact)
        thr_copy_s2t = tiled_copy_s2t.get_slice(0)
        # ((ATOM_V, REST_V), Rest_Tiler, MMA_MN, MMA_K, STAGE)
        tCsSF_compact_s2t_ = thr_copy_s2t.partition_S(tCsSF_compact)
        # ((ATOM_V, REST_V), Rest_Tiler, MMA_MN, MMA_K, STAGE)
        tCsSF_compact_s2t = tcgen05.get_s2t_smem_desc_tensor(tiled_copy_s2t, tCsSF_compact_s2t_)
        # ((ATOM_V, REST_V), Rest_Tiler, MMA_MN, MMA_K)
        tCtSF_compact_s2t = thr_copy_s2t.partition_D(tCtSF_compact)
        return tiled_copy_s2t, tCsSF_compact_s2t, tCtSF_compact_s2t

    def epilog_tmem_copy_and_partition(
        self,
        tidx: Int32,
        tAcc: cute.Tensor,
        epi_tile: cute.Tile,
        use_2cta_instrs: Union[Boolean, bool],
    ) -> Tuple[cute.TiledCopy, cute.Tensor, cute.Tensor]:
        """
        Make tiledCopy for tensor memory load, then use it to partition tensor memory (source) and register array (destination).

        :param tidx: The thread index in epilogue warp groups
        :type tidx: Int32
        :param tAcc: The accumulator tensor to be copied and partitioned
        :type tAcc: cute.Tensor
        :param epi_tile: The epilogue tiler
        :type epi_tile: cute.Tile
        :param use_2cta_instrs: Whether use_2cta_instrs is enabled
        :type use_2cta_instrs: bool

        :return: A tuple containing (tiled_copy_t2r, tTR_tAcc, tTR_rAcc) where:
            - tiled_copy_t2r: The tiled copy operation for tmem to register copy(t2r)
            - tTR_tAcc: The partitioned accumulator tensor
            - tTR_rAcc: The accumulated tensor in register used to hold t2r results
        :rtype: Tuple[cute.TiledCopy, cute.Tensor, cute.Tensor]
        """
        # Make tiledCopy for tensor memory load
        copy_atom_t2r = sm100_utils.get_tmem_load_op(
            self.cta_tile_shape_mnk,
            self.d_layout if self.d_layout is not None else LayoutEnum.ROW_MAJOR,
            self.d_dtype if self.d_dtype is not None else cutlass.BFloat16,
            self.acc_dtype,
            epi_tile,
            use_2cta_instrs,
        )
        # (EPI_TILE_M, EPI_TILE_N, EPI_M, EPI_N, STAGE)
        tAcc_epi = cute.flat_divide(tAcc[((None, None), 0, 0, None)], epi_tile)
        # (EPI_TILE_M, EPI_TILE_N)
        tiled_copy_t2r = tcgen05.make_tmem_copy(copy_atom_t2r, tAcc_epi[(None, None, 0, 0, 0)])

        thr_copy_t2r = tiled_copy_t2r.get_slice(tidx)
        # (T2R, T2R_M, T2R_N, EPI_M, EPI_M, STAGE)
        tTR_tAcc = thr_copy_t2r.partition_S(tAcc_epi)

        cAcc = cute.make_identity_tensor((self.cta_tile_shape_mnk[0], self.cta_tile_shape_mnk[1]))
        # (EPI_TILE_M, EPI_TILE_N, EPI_M, EPI_N)
        cAcc_epi = cute.flat_divide(cAcc, epi_tile)
        # (T2R, T2R_M, T2R_N, EPI_M, EPI_N)
        tTR_cAcc = thr_copy_t2r.partition_D(cAcc_epi)
        # (T2R, T2R_M, T2R_N)
        tTR_rAcc = cute.make_fragment(tTR_cAcc[None, None, None, 0, 0].shape, self.acc_dtype)
        return tiled_copy_t2r, tTR_tAcc, tTR_rAcc

    def epilog_smem_store_and_partition(
        self,
        tiled_copy_t2r: cute.TiledCopy,
        d_layout: Optional[LayoutEnum],
        dtype: Optional[Type[cutlass.Numeric]],
        tTR_rD: cute.Tensor,
        sD: cute.Tensor,
        tidx: Int32,
    ) -> Tuple[cute.TiledCopy, cute.Tensor, cute.Tensor]:
        """
        Make tiledCopy for shared memory store, then use it to partition register array (source) and shared memory (destination).

        :param tiled_copy_t2r: The tiled copy operation for tmem to register copy(t2r)
        :type tiled_copy_t2r: cute.TiledCopy
        :param tTR_rD: The partitioned accumulator tensor
        :type tTR_rD: cute.Tensor
        :param tidx: The thread index in epilogue warp groups
        :type tidx: Int32
        :param sD: The shared memory tensor to be copied and partitioned
        :type sD: cute.Tensor
        :type sepi: cute.Tensor

        :return: A tuple containing (tiled_copy_r2s, tRS_rD, tRS_sD) where:
            - tiled_copy_r2s: The tiled copy operation for register to smem copy(r2s)
            - tRS_rD: The partitioned tensor C (register source)
            - tRS_sD: The partitioned tensor C (smem destination)
        :rtype: Tuple[cute.TiledCopy, cute.Tensor, cute.Tensor]
        """
        copy_atom_r2s = sm100_utils.get_smem_store_op(
            d_layout if d_layout is not None else LayoutEnum.ROW_MAJOR,
            dtype if dtype is not None else cutlass.BFloat16,
            self.acc_dtype,
            tiled_copy_t2r,
        )
        tiled_copy_r2s = cute.make_tiled_copy_D(copy_atom_r2s, tiled_copy_t2r)
        # (R2S, R2S_M, R2S_N, PIPE_D)
        thr_copy_r2s = tiled_copy_r2s.get_slice(tidx)
        tRS_sD = thr_copy_r2s.partition_D(sD) if sD is not None else None
        # (R2S, R2S_M, R2S_N)
        tRS_rD = tiled_copy_r2s.retile(tTR_rD)
        return tiled_copy_r2s, tRS_rD, tRS_sD

    def epilog_smem_load_and_partition(
        self,
        tiled_copy_t2r: cute.TiledCopy,
        c_layout: LayoutEnum,
        dtype: Type[cutlass.Numeric],
        # tTR_rC: cute.Tensor,
        sC: cute.Tensor,
        tRS_rD_layout: cutlass.Layout,
        tidx: Int32,
    ) -> Tuple[cute.TiledCopy, cute.Tensor, cute.Tensor]:
        copy_atom_r2s = sm100_utils.get_smem_store_op(
            c_layout, dtype, self.acc_dtype, tiled_copy_t2r
        )
        store_op = copy_atom_r2s.op
        # m8n8 16-bit path
        if isinstance(store_op, StMatrix8x8x16bOp):
            op = LdMatrix8x8x16bOp(num_matrices=store_op.num_matrices, transpose=store_op.transpose)
        # m16n8 8-bit store -> m16n16 8-bit load
        elif isinstance(store_op, StMatrix16x8x8bOp) and store_op.num_matrices in [2, 4]:
            # transpose=True is enforced by the class
            op = LdMatrix16x16x8bOp(num_matrices=store_op.num_matrices // 2)
        else:
            op = cute.nvgpu.CopyUniversalOp()
        copy_atom_s2r = cute.make_copy_atom(op, dtype)
        tiled_copy_s2r = cute.make_tiled_copy_D(copy_atom_s2r, tiled_copy_t2r)
        thr_copy_s2r = tiled_copy_s2r.get_slice(tidx)
        # (R2S, R2S_M, R2S_N, PIPE_D)
        tSR_sC = thr_copy_s2r.partition_S(sC)
        tRS_rC = cute.make_fragment(tRS_rD_layout, dtype)
        # (R2S, R2S_M, R2S_N)
        tSR_rC = tiled_copy_s2r.retile(tRS_rC)
        return tiled_copy_s2r, tRS_rC, tSR_rC, tSR_sC

    @cute.jit
    def make_ab_pipeline(
        self,
        tiled_mma: cute.TiledMma,
        cluster_layout_vmnk: cute.Layout,
        ab_pipeline_mbar_ptr: cute.Pointer,
        is_leader_cta: Boolean,
    ) -> pipeline.PipelineAsync:
        # If gather_A and use_2cta_instrs, the cp.async for the non-leader CTA will
        # arrive at an mbarrier on the non-leader CTA side, then the mma warp of the non-leader
        # CTA will wait for that then arrive at the mbarrier on the leader CTA.
        # The producer count for the leader CTA is 1 (TMA) + num_cpasync_threads
        # + 1 (from non-leader CTA).
        # The producer count for the non-leader CTA is num_cpasync_threads
        # (TMA doesn't arrive there).
        if const_expr(not self.gather_A):
            producer_cnt = 1
        else:
            producer_cnt = (self.num_ab_load_warps - 1) * 32 + (
                1 if const_expr(not self.use_2cta_instrs) else 2
            )
        ab_pipeline_producer_group = pipeline.CooperativeGroup(pipeline.Agent.Thread, producer_cnt)
        # Each warp will contribute to the arrive count with the number of mcast size
        mcast_size = self.num_mcast_ctas_a + self.num_mcast_ctas_b - 1
        consumer_arrive_cnt = mcast_size
        ab_pipeline_consumer_group = pipeline.CooperativeGroup(
            pipeline.Agent.Thread, consumer_arrive_cnt
        )
        if const_expr(not self.gather_A):
            pipeline_ab = pipeline.PipelineTmaUmma.create(
                barrier_storage=ab_pipeline_mbar_ptr,
                num_stages=self.ab_stage,
                producer_group=ab_pipeline_producer_group,
                consumer_group=ab_pipeline_consumer_group,
                tx_count=self.num_tma_load_bytes,
                cta_layout_vmnk=cluster_layout_vmnk,
            )
        else:
            pipeline_ab = PipelineTmaCpAsyncUmma.create(
                barrier_storage=ab_pipeline_mbar_ptr,
                num_stages=self.ab_stage,
                producer_group=ab_pipeline_producer_group,
                consumer_group=ab_pipeline_consumer_group,
                tx_count=self.num_tma_load_bytes,
                cta_layout_vmnk=cluster_layout_vmnk,
                producer_drop_count=None
                if not self.use_2cta_instrs
                else (2 if not is_leader_cta else 0),
            )
        return pipeline_ab

    def make_acc_pipeline(
        self, cluster_layout_vmnk: cute.Layout, acc_pipeline_mbar_ptr: cute.Pointer
    ) -> pipeline.PipelineAsync:
        acc_pipeline_producer_group = pipeline.CooperativeGroup(pipeline.Agent.Thread)
        num_acc_consumer_threads = self.num_epi_warps * (2 if self.use_2cta_instrs else 1)
        acc_pipeline_consumer_group = pipeline.CooperativeGroup(
            pipeline.Agent.Thread, num_acc_consumer_threads
        )
        return pipeline.PipelineUmmaAsync.create(
            barrier_storage=acc_pipeline_mbar_ptr,
            num_stages=self.num_acc_stage,
            producer_group=acc_pipeline_producer_group,
            consumer_group=acc_pipeline_consumer_group,
            cta_layout_vmnk=cluster_layout_vmnk,
        )

    def make_sched_pipeline(
        self,
        cluster_layout_mnk: cute.Layout,
        sched_pipeline_mbar_ptr: cute.Pointer,
        has_C: bool = False,
    ) -> pipeline.PipelineAsync:
        # Threads/warps participating in this pipeline
        sched_pipeline_producer_group = pipeline.CooperativeGroup(pipeline.Agent.Thread)
        cluster_size = cute.size(cluster_layout_mnk)
        # Each warp that are not the scheduler warp will contribute 1 to the arrive count
        warps_per_cta = self.num_ab_load_warps + len(
            (self.mma_warp_id, *self.epilog_warp_id, self.scheduler_warp_id)
        )
        if has_C:
            warps_per_cta += 1
        consumer_arrive_cnt = warps_per_cta * cluster_size - 1
        sched_pipeline_consumer_group = pipeline.CooperativeGroup(
            pipeline.Agent.Thread, consumer_arrive_cnt
        )
        return pipeline.PipelineAsync.create(
            barrier_storage=sched_pipeline_mbar_ptr,
            num_stages=self.sched_stage,
            producer_group=sched_pipeline_producer_group,
            consumer_group=sched_pipeline_consumer_group,
            # If there's cluster, the consumers must arrive at the mbar of CTA 0 in the cluster.
            consumer_mask=None if const_expr(cluster_size == 1) else 0,
        )

    @cute.jit
    def make_a_prefetch_pipeline(
        self, a_prefetch_pipeline_mbar_ptr: cute.Pointer
    ) -> pipeline.PipelineAsync:
        producer_cnt = 32
        a_prefetch_producer_group = pipeline.CooperativeGroup(
            pipeline.Agent.Thread, producer_cnt, alignment=producer_cnt
        )
        consumer_arrive_cnt = self.num_ab_load_warps - 1
        a_prefetch_consumer_group = pipeline.CooperativeGroup(
            pipeline.Agent.Thread, consumer_arrive_cnt
        )
        return pipeline.PipelineCpAsync.create(
            barrier_storage=a_prefetch_pipeline_mbar_ptr,
            num_stages=self.a_prefetch_stage,
            producer_group=a_prefetch_producer_group,
            consumer_group=a_prefetch_consumer_group,
        )

    @classmethod
    def _compute_stages(
        cls,
        tiled_mma: cute.TiledMma,
        mma_tiler_mnk: Tuple[int, int, int],
        cta_tile_shape_mnk: Tuple[int, int, int],
        epi_tile: cute.Tile,
        a_dtype: Type[cutlass.Numeric],
        b_dtype: Type[cutlass.Numeric],
        sf_dtype: Optional[Type[cutlass.Numeric]],
        sf_vec_size: Optional[int],
        d_dtype: Optional[Type[cutlass.Numeric]],
        c_dtype: Optional[Type[cutlass.Numeric]],
        d_layout: Optional[LayoutEnum],
        c_layout: Optional[LayoutEnum],
        epilogue_args: EpilogueArguments,
        prefetch_A_idx: Literal[None, "varlen_m", "varlen_k"],
        smem_capacity: int,
        occupancy: int,
    ) -> Tuple[int, int, int]:
        """Computes the number of stages for A/B/C operands based on heuristics.

        :param tiled_mma: The tiled MMA object defining the core computation.
        :type tiled_mma: cute.TiledMma
        :param mma_tiler_mnk: The shape (M, N, K) of the MMA tiler.
        :type mma_tiler_mnk: tuple[int, int, int]
        :param a_dtype: Data type of operand A.
        :type a_dtype: type[cutlass.Numeric]
        :param b_dtype: Data type of operand B.
        :type b_dtype: type[cutlass.Numeric]
        :param epi_tile: The epilogue tile shape.
        :type epi_tile: cute.Tile
        :param d_dtype: Data type of operand C (output).
        :type d_dtype: type[cutlass.Numeric]
        :param d_layout: Layout enum of operand D.
        :type d_layout: LayoutEnum
        :param smem_capacity: Total available shared memory capacity in bytes.
        :type smem_capacity: int
        :param occupancy: Target number of CTAs per SM (occupancy).
        :type occupancy: int

        :return: A tuple containing the computed number of stages for:
                 (ACC stages, A/B operand stages, C stages)
        :rtype: tuple[int, int, int]
        """
        blockscaled = sf_dtype is not None
        # Default ACC stages
        if const_expr(not blockscaled):
            num_acc_stage = 2
        else:
            num_acc_stage = 1 if mma_tiler_mnk[1] == 256 else 2

        # Default D stages
        epi_stage = 4 if cute.size(epi_tile[1]) <= 16 else 2
        epi_c_stage = 0 if c_dtype is None else (4 if cute.size(epi_tile[1]) <= 16 else 2)

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
        d_smem_layout_staged_one = (
            sm100_utils.make_smem_layout_epi(d_dtype, d_layout, epi_tile, 1)
            if d_dtype is not None
            else None
        )
        c_smem_layout_staged_one = (
            sm100_utils.make_smem_layout_epi(c_dtype, c_layout, epi_tile, 1)
            if c_dtype is not None
            else None
        )
        if const_expr(blockscaled):
            sfa_smem_layout_staged_one = blockscaled_utils.make_smem_layout_sfa(
                tiled_mma,
                mma_tiler_mnk,
                sf_vec_size,
                1,  # a tmp 1 stage is provided
            )
            sfb_smem_layout_staged_one = blockscaled_utils.make_smem_layout_sfb(
                tiled_mma,
                mma_tiler_mnk,
                sf_vec_size,
                1,  # a tmp 1 stage is provided
            )

        ab_bytes_per_stage = cute.size_in_bytes(
            a_dtype, a_smem_layout_staged_one
        ) + cute.size_in_bytes(b_dtype, b_smem_layout_staged_one)
        if const_expr(prefetch_A_idx == "varlen_k"):  # Need smem to prefetch A indices
            ab_bytes_per_stage += Int32.width // 8 * cta_tile_shape_mnk[2]
        if const_expr(blockscaled):
            ab_bytes_per_stage += cute.size_in_bytes(
                sf_dtype, sfa_smem_layout_staged_one
            ) + cute.size_in_bytes(sf_dtype, sfb_smem_layout_staged_one)
        mbar_helpers_bytes = 1024
        if const_expr(prefetch_A_idx == "varlen_m"):
            mbar_helpers_bytes += Int32.width // 8 * cta_tile_shape_mnk[0] * 2
        d_bytes_per_stage = (
            cute.size_in_bytes(d_dtype, d_smem_layout_staged_one) if d_dtype is not None else 0
        )
        epi_bytes_per_stage = d_bytes_per_stage + cls.epi_smem_bytes_per_stage(
            epilogue_args, cta_tile_shape_mnk, epi_tile
        )
        epi_bytes = epi_bytes_per_stage * epi_stage
        if const_expr(c_dtype is not None):
            c_bytes_per_stage = cute.size_in_bytes(c_dtype, c_smem_layout_staged_one)
            epi_bytes += c_bytes_per_stage * epi_c_stage

        # Calculate A/B/SFA/SFB stages:
        # Start with total smem per CTA (capacity / occupancy)
        # Subtract reserved bytes and initial C stages bytes
        # Divide remaining by bytes needed per A/B/SFA/SFB stage
        remaining_bytes = smem_capacity // occupancy - mbar_helpers_bytes - epi_bytes
        ab_stage = remaining_bytes // ab_bytes_per_stage

        # Refine epilogue stages:
        # Calculate remaining smem after allocating for A/B stages and reserved bytes
        # Add remaining unused smem to epilogue
        epi_stage += (remaining_bytes - ab_bytes_per_stage * ab_stage) // (epi_bytes_per_stage)
        return num_acc_stage, ab_stage, epi_stage, epi_c_stage

    @staticmethod
    def _compute_num_tmem_alloc_cols(
        tiled_mma: cute.TiledMma,
        mma_tiler: Tuple[int, int, int],
        num_acc_stage: int,
    ) -> int:
        """
        Compute the number of tensor memory allocation columns.

        :param tiled_mma: The tiled MMA object defining the core computation.
        :type tiled_mma: cute.TiledMma
        :param mma_tiler: The shape (M, N, K) of the MMA tile.
        :type mma_tiler: tuple[int, int, int]
        :param num_acc_stage: The stage of the accumulator tensor.
        :type num_acc_stage: int

        :return: The number of tensor memory allocation columns.
        :rtype: int
        """
        acc_shape = tiled_mma.partition_shape_C(mma_tiler[:2])
        tCtAcc_fake = tiled_mma.make_fragment_C(cute.append(acc_shape, num_acc_stage))
        num_tmem_alloc_cols = cutlass.utils.get_num_tmem_alloc_cols(tCtAcc_fake)
        return num_tmem_alloc_cols

    @staticmethod
    def is_valid_dtypes(
        a_dtype: Type[cutlass.Numeric],
        b_dtype: Type[cutlass.Numeric],
        acc_dtype: Type[cutlass.Numeric],
        d_dtype: Optional[Type[cutlass.Numeric]],
        a_major: str,
        b_major: str,
    ) -> bool:
        """
        Check if the dtypes are valid

        :param ab_dtype: The data type of the A and B operands
        :type ab_dtype: Type[cutlass.Numeric]
        :param acc_dtype: The data type of the accumulator
        :type acc_dtype: Type[cutlass.Numeric]
        :param d_dtype: The data type of the output tensor
        :type d_dtype: Type[cutlass.Numeric]

        :return: True if the dtypes are valid, False otherwise
        :rtype: bool
        """
        is_valid = True
        if b_dtype != a_dtype:
            is_valid = False
        ab_dtype = a_dtype
        if ab_dtype not in {
            cutlass.Float16,
            cutlass.BFloat16,
            cutlass.TFloat32,
            cutlass.Uint8,
            cutlass.Int8,
            cutlass.Float8E4M3FN,
            cutlass.Float8E5M2,
        }:
            is_valid = False
        if (
            acc_dtype not in {Float32, cutlass.Float16, Int32}
            or acc_dtype == cutlass.Float16
            and ab_dtype not in {cutlass.Float16, cutlass.Float8E4M3FN, cutlass.Float8E5M2}
            or acc_dtype == Int32
            and ab_dtype not in {cutlass.Uint8, cutlass.Int8}
        ):
            is_valid = False
        if d_dtype is not None and (
            acc_dtype == Float32
            and d_dtype
            not in {
                Float32,
                cutlass.Float16,
                cutlass.BFloat16,
                cutlass.Float8E4M3FN,
                cutlass.Float8E5M2,
                Int32,
                cutlass.Int8,
                cutlass.Uint8,
            }
            or acc_dtype == cutlass.Float16
            and d_dtype
            not in {
                cutlass.BFloat16,
                cutlass.Float16,
            }
            or acc_dtype == Int32
            and d_dtype
            not in {
                cutlass.BFloat16,
                cutlass.Float16,
                Float32,
                Int32,
                cutlass.Int8,
                cutlass.Uint8,
            }
        ):
            is_valid = False
        if ab_dtype is cutlass.Float4E2M1FN and not (a_major == "k" and b_major == "k"):
            is_valid = False
        return is_valid

    @staticmethod
    def is_valid_dtypes_and_scale_factor_vec_size(
        ab_dtype: Type[cutlass.Numeric],
        sf_dtype: Type[cutlass.Numeric],
        sf_vec_size: int,
        d_dtype: Type[cutlass.Numeric],
    ) -> bool:
        """
        Check if the dtypes and sf_vec_size are valid combinations

        :param ab_dtype: The data type of the A and B operands
        :type ab_dtype: Type[cutlass.Numeric]
        :param sf_dtype: The data type of the scale factor
        :type sf_dtype: Type[cutlass.Numeric]
        :param sf_vec_size: The vector size of the scale factor
        :type sf_vec_size: int
        :param d_dtype: The data type of the output tensor
        :type d_dtype: Type[cutlass.Numeric]

        :return: True if the dtypes and sf_vec_size are valid, False otherwise
        :rtype: bool
        """
        is_valid = True

        # Check valid ab_dtype
        if ab_dtype not in {cutlass.Float4E2M1FN, cutlass.Float8E5M2, cutlass.Float8E4M3FN}:
            is_valid = False

        # Check valid sf_vec_size
        if sf_vec_size not in {16, 32}:
            is_valid = False

        # Check valid sf_dtype
        if sf_dtype not in {cutlass.Float8E8M0FNU, cutlass.Float8E4M3FN}:
            is_valid = False

        # Check valid sf_dtype and sf_vec_size combinations
        if sf_dtype == cutlass.Float8E4M3FN and sf_vec_size == 32:
            is_valid = False
        if ab_dtype in {cutlass.Float8E5M2, cutlass.Float8E4M3FN} and sf_vec_size == 16:
            is_valid = False

        # Check valid d_dtype
        if d_dtype not in {
            Float32,
            cutlass.Float16,
            cutlass.BFloat16,
            cutlass.Float8E5M2,
            cutlass.Float8E4M3FN,
        }:
            is_valid = False

        return is_valid

    @staticmethod
    def is_valid_mma_tiler_and_cluster_shape(
        mma_tiler_mn: Tuple[int, int],
        cluster_shape_mn: Tuple[int, int],
        blockscaled: bool,
    ) -> bool:
        """
        Check if the mma tiler and cluster shape are valid

        :param mma_tiler_mn: The (M, N) shape of the MMA instruction tiler
        :type mma_tiler_mn: Tuple[int, int]
        :param cluster_shape_mn: The (ClusterM, ClusterN) shape of the CTA cluster
        :type cluster_shape_mn: Tuple[int, int]

        :return: True if the mma tiler and cluster shape are valid, False otherwise
        :rtype: bool
        """
        is_valid = True
        # Skip invalid mma tile shape
        if mma_tiler_mn[0] not in [64, 128, 256]:
            is_valid = False
        if not blockscaled:
            if mma_tiler_mn[1] not in range(32, 257, 32):
                is_valid = False
        else:
            if mma_tiler_mn[1] not in [128, 256]:
                is_valid = False
        # Skip invalid cluster shape
        is_power_of_2 = lambda x: x > 0 and (x & (x - 1)) == 0
        if (
            cluster_shape_mn[0] * cluster_shape_mn[1] > 16
            or cluster_shape_mn[0] <= 0
            or cluster_shape_mn[1] <= 0
            or not is_power_of_2(cluster_shape_mn[0])
            or not is_power_of_2(cluster_shape_mn[1])
        ):
            is_valid = False
        if blockscaled:
            # Special cluster shape check for scale factor multicasts.
            # Due to limited size of scale factors, we can't multicast among more than 4 CTAs.
            if cluster_shape_mn[0] > 4 or cluster_shape_mn[1] > 4:
                is_valid = False
        return is_valid

    @staticmethod
    def is_valid_tensor_alignment(
        m: int,
        n: int,
        k: int,
        l: int,
        ab_dtype: Type[cutlass.Numeric],
        d_dtype: Type[cutlass.Numeric],
        a_major: str,
        b_major: str,
        d_major: str,
    ) -> bool:
        """
        Check if the tensor alignment is valid

        :param m: The number of rows in the A tensor
        :type m: int
        :param n: The number of columns in the B tensor
        :type n: int
        :param k: The number of columns in the A tensor
        :type k: int
        :param l: The number of columns in the C tensor
        :type l: int
        :param ab_dtype: The data type of the A and B operands
        :type ab_dtype: Type[cutlass.Numeric]
        :param d_dtype: The data type of the output tensor
        :type d_dtype: Type[cutlass.Numeric]
        :param a_major: The major axis of the A tensor
        :type a_major: str
        :param b_major: The major axis of the B tensor
        :type b_major: str
        :param d_major: The major axis of the C tensor
        :type d_major: str

        :return: True if the problem shape is valid, False otherwise
        :rtype: bool
        """
        is_valid = True

        def check_contigous_16B_alignment(dtype, is_mode0_major, tensor_shape):
            major_mode_idx = 0 if is_mode0_major else 1
            num_major_elements = tensor_shape[major_mode_idx]
            num_contiguous_elements = 16 * 8 // dtype.width
            return num_major_elements % num_contiguous_elements == 0

        if (
            not check_contigous_16B_alignment(ab_dtype, a_major == "m", (m, k, l))
            or not check_contigous_16B_alignment(ab_dtype, b_major == "n", (n, k, l))
            or not check_contigous_16B_alignment(d_dtype, d_major == "m", (m, n, l))
        ):
            is_valid = False
        return is_valid

    @staticmethod
    def can_implement(
        ab_dtype: Type[cutlass.Numeric],
        acc_dtype: Type[cutlass.Numeric],
        d_dtype: Type[cutlass.Numeric],
        mma_tiler_mn: Tuple[int, int],
        cluster_shape_mn: Tuple[int, int],
        m: int,
        n: int,
        k: int,
        l: int,
        a_major: str,
        b_major: str,
        d_major: str,
    ) -> bool:
        """
        Check if the gemm can be implemented

        :param ab_dtype: The data type of the A and B operands
        :type ab_dtype: Type[cutlass.Numeric]
        :param acc_dtype: The data type of the accumulator
        :type acc_dtype: Type[cutlass.Numeric]
        :param d_dtype: The data type of the output tensor
        :type d_dtype: Type[cutlass.Numeric]
        :param mma_tiler_mn: The (M, N) shape of the MMA instruction tiler
        :type mma_tiler_mn: Tuple[int, int]
        :param cluster_shape_mn: The (ClusterM, ClusterN) shape of the CTA cluster
        :type cluster_shape_mn: Tuple[int, int]
        :param m: The number of rows in the A tensor
        :type m: int
        :param n: The number of columns in the B tensor
        :type n: int
        :param k: The number of columns in the A tensor
        :type k: int
        :param l: The number of columns in the C tensor
        :type l: int
        :param a_major: The major axis of the A tensor
        :type a_major: str
        :param b_major: The major axis of the B tensor
        :type b_major: str
        :param d_major: The major axis of the C tensor
        :type d_major: str

        :return: True if the gemm can be implemented, False otherwise
        :rtype: bool
        """
        can_implement = True
        # Skip unsupported types
        if not GemmSm100.is_valid_dtypes(ab_dtype, ab_dtype, acc_dtype, d_dtype, a_major, b_major):
            can_implement = False
        # Skip invalid mma tile shape and cluster shape
        if not GemmSm100.is_valid_mma_tiler_and_cluster_shape(
            mma_tiler_mn, cluster_shape_mn, blockscaled=False
        ):
            can_implement = False
        # Skip illegal problem shape for load/store alignment
        if not GemmSm100.is_valid_tensor_alignment(
            m, n, k, l, ab_dtype, d_dtype, a_major, b_major, d_major
        ):
            can_implement = False
        return can_implement


def run(
    mnkl: Tuple[int, int, int, int],
    ab_dtype: Type[cutlass.Numeric],
    d_dtype: Type[cutlass.Numeric],
    c_dtype: Optional[Type[cutlass.Numeric]],
    acc_dtype: Type[cutlass.Numeric],
    a_major: str,
    b_major: str,
    d_major: str,
    c_major: str,
    mma_tiler_mn: Tuple[int, int] = (256, 256),
    cluster_shape_mn: Tuple[int, int] = (2, 1),
    tolerance: float = 1e-01,
    warmup_iterations: int = 0,
    iterations: int = 1,
    skip_ref_check: bool = False,
    dynamic_persistent: bool = False,
    **kwargs,
):
    """Execute a persistent batched dense GEMM operation on Blackwell architecture with performance benchmarking.

    This function prepares input tensors, configures and launches the persistent GEMM kernel,
    optionally performs reference validation, and benchmarks the execution performance.

    :param mnkl: Problem size (M, N, K, L)
    :type mnkl: Tuple[int, int, int, int]
    :param ab_dtype: Data type for input tensors A and B
    :type ab_dtype: Type[cutlass.Numeric]
    :param d_dtype: Data type for output tensor C
    :type d_dtype: Type[cutlass.Numeric]
    :param acc_dtype: Data type for accumulation during matrix multiplication
    :type acc_dtype: Type[cutlass.Numeric]
    :param a_major/b_major/d_major: Memory layout of tensor A/B/C
    :type a_major/b_major/d_major: str
    :param mma_tiler_mn: MMA tiling size. If not specified in the decorator parameters, the autotuner will use the
        default value of (256, 256). Otherwise, the autotuner will use the value specified in the decorator parameters.
    :type mma_tiler_mn: Tuple[int, int], optional
    :param cluster_shape_mn: Cluster shape. If not specified in the decorator parameters, the autotuner will use the
        default value of (2, 1). Otherwise, the autotuner will use the value specified in the decorator parameters.
    :type cluster_shape_mn: Tuple[int, int], optional
    :param tolerance: Tolerance value for reference validation comparison, defaults to 1e-01
    :type tolerance: float, optional
    :param warmup_iterations: Number of warmup iterations before benchmarking, defaults to 0
    :type warmup_iterations: int, optional
    :param iterations: Number of benchmark iterations to run, defaults to 1
    :type iterations: int, optional
    :param skip_ref_check: Whether to skip reference result validation, defaults to False
    :type skip_ref_check: bool, optional
    :raises RuntimeError: If CUDA GPU is not available
    :raises ValueError: If the configuration is invalid or unsupported by the kernel
    :return: Execution time of the GEMM kernel
    :rtype: float
    """
    print("Running Blackwell Persistent Dense GEMM test with:")
    print(f"mnkl: {mnkl}")
    print(f"AB dtype: {ab_dtype}, C dtype: {d_dtype}, Acc dtype: {acc_dtype}")
    print(f"Matrix majors - A: {a_major}, B: {b_major}, C: {d_major}")
    print(f"Mma Tiler (M, N): {mma_tiler_mn}, Cluster Shape (M, N): {cluster_shape_mn}")
    print(f"Tolerance: {tolerance}")
    print(f"Warmup iterations: {warmup_iterations}")
    print(f"Iterations: {iterations}")
    print(f"Skip reference checking: {skip_ref_check}")

    assert not dynamic_persistent, "Dynamic persistent mode is not supported yet."

    # Unpack parameters
    m, n, k, l = mnkl

    # Skip unsupported testcase
    if not GemmSm100.can_implement(
        ab_dtype,
        acc_dtype,
        d_dtype,
        mma_tiler_mn,
        cluster_shape_mn,
        m,
        n,
        k,
        l,
        a_major,
        b_major,
        d_major,
    ):
        raise TypeError(
            f"Unsupported testcase {ab_dtype}, {acc_dtype}, {d_dtype}, {mma_tiler_mn}, {cluster_shape_mn}, {m}, {n}, {k}, {l}, {a_major}, {b_major}, {d_major}"
        )

    if not torch.cuda.is_available():
        raise RuntimeError("GPU is required to run this example!")

    torch.manual_seed(1111)

    # Create and permute tensor A/B/C
    def create_and_permute_tensor(l, mode0, mode1, is_mode0_major, dtype, is_dynamic_layout=True):
        # is_mode0_major: (l, mode1, mode0) -> (mode0, mode1, l)
        # else: (l, mode0, mode1) -> (mode0, mode1, l)
        shape = (l, mode1, mode0) if is_mode0_major else (l, mode0, mode1)
        permute_order = (2, 1, 0) if is_mode0_major else (1, 2, 0)
        is_unsigned = dtype in {cutlass.Uint8}
        # Temporarily use uint8 as torch does not support fp8 type
        torch_dtype = cutlass_torch.dtype(dtype)
        gen_dtype = (
            torch_dtype
            if dtype not in {cutlass.Float8E5M2, cutlass.Float8E4M3FN}
            else torch.bfloat16
        )

        # Create dtype torch tensor (cpu)
        torch_tensor_cpu = cutlass_torch.create_and_permute_torch_tensor(
            shape,
            gen_dtype,
            permute_order=permute_order,
            # init_type=cutlass.torch.TensorInitType.RANDOM,
            # init_config=cutlass.torch.RandomInitConfig(
            #     min_val=0 if is_unsigned else -2, max_val=4 if is_unsigned else 2
            # ),
            init_type=cutlass.torch.TensorInitType.GAUSSIAN,
            init_config=cutlass.torch.GaussianInitConfig(std=k ** (-0.5), scale=1),
        ).to(torch_dtype)
        # Create dtype torch tensor (gpu)
        torch_tensor = torch_tensor_cpu.cuda()

        # Create f32 torch tensor (cpu)
        f32_torch_tensor = torch_tensor_cpu.to(dtype=torch.float32)

        # Create dtype cute tensor (gpu)
        torch_tensor_view = (
            torch_tensor
            if dtype not in {cutlass.Float8E5M2, cutlass.Float8E4M3FN}
            else torch_tensor.view(torch.uint8)
        )
        cute_tensor = from_dlpack(torch_tensor_view, assumed_align=16)
        cute_tensor.element_type = dtype
        if is_dynamic_layout:
            cute_tensor = cute_tensor.mark_layout_dynamic(leading_dim=(0 if is_mode0_major else 1))
        cute_tensor = cutlass_torch.convert_cute_tensor(
            f32_torch_tensor,
            cute_tensor,
            dtype,
            is_dynamic_layout=is_dynamic_layout,
        )

        return f32_torch_tensor, cute_tensor, torch_tensor, torch_tensor_cpu

    a_ref, mA, a_torch, a_torch_cpu = create_and_permute_tensor(
        l, m, k, a_major == "m", ab_dtype, is_dynamic_layout=True
    )
    b_ref, mB, b_torch, b_torch_cpu = create_and_permute_tensor(
        l, n, k, b_major == "n", ab_dtype, is_dynamic_layout=True
    )
    _, mD, d_torch, d_torch_cpu = create_and_permute_tensor(
        l, m, n, d_major == "m", d_dtype, is_dynamic_layout=True
    )
    if c_dtype is not None:
        c, mC, c_torch, d_torch_cpu = create_and_permute_tensor(l, m, n, c_major == "m", c_dtype)
    else:
        c, mC, c_torch = None, None, None

    # Configure gemm kernel
    cluster_shape_mnk = (*cluster_shape_mn, 1)
    gemm = GemmSm100(acc_dtype, ab_dtype, mma_tiler_mn, cluster_shape_mnk)

    # Compute max active clusters on current device
    hardware_info = cutlass.utils.HardwareInfo()
    max_active_clusters = hardware_info.get_max_active_clusters(
        cluster_shape_mn[0] * cluster_shape_mn[1]
    )
    if dynamic_persistent:
        tile_count_semaphore = torch.zeros(1, dtype=torch.int32, device="cuda")
    else:
        tile_count_semaphore = None

    scheduler_args = TileSchedulerOptions(
        Int32(max_active_clusters),
        tile_count_semaphore=make_ptr(
            Int32, tile_count_semaphore.data_ptr(), cute.AddressSpace.gmem, assumed_align=4
        )
        if tile_count_semaphore is not None
        else None,
    )
    epi_args = gemm.EpilogueArguments()
    varlen_args = VarlenArguments()

    # Get current CUDA stream from PyTorch
    torch_stream = torch.cuda.current_stream()
    # Get the raw stream pointer as a CUstream
    current_stream = cuda.CUstream(torch_stream.cuda_stream)
    # Compile gemm kernel
    compiled_gemm = cute.compile(
        gemm,
        mA,
        mB,
        mD,
        mC,
        epi_args,
        scheduler_args,
        varlen_args,
        current_stream,
    )

    if not skip_ref_check:
        compiled_gemm(mA, mB, mD, mC, epi_args, scheduler_args, varlen_args, current_stream)
        if ab_dtype in {
            cutlass.Int8,
            cutlass.Uint8,
            cutlass.Float8E4M3FN,
            cutlass.Float8E5M2,
        }:
            ref = torch.einsum("mkl,nkl->mnl", a_ref.cpu(), b_ref.cpu())
        else:
            ref = torch.einsum("mkl,nkl->mnl", a_ref, b_ref)
        if c is not None:
            ref = ref + c
        ref = ref.cpu()

        # Copy gpu result back
        gpu_d = d_torch.cpu()

        # Convert ref to c_type
        if d_dtype == Float32:
            ref_d = ref
        elif d_dtype in {cutlass.Float8E5M2, cutlass.Float8E4M3FN}:
            # m major: (l, n, m) -> (m, n, l)
            # n major: (l, m, n) -> (m, n, l)
            permute_order = (1, 2, 0) if d_major == "n" else (2, 1, 0)
            shape = (l, m, n) if d_major == "n" else (l, n, m)
            f8_torch_tensor = cutlass_torch.create_and_permute_torch_tensor(
                shape,
                torch.uint8,
                permute_order=permute_order,
                init_type=cutlass_torch.TensorInitType.SKIP,
            ).cuda()
            # Create dtype cute tensor (gpu)
            ref_d_tensor = from_dlpack(f8_torch_tensor, assumed_align=16).mark_layout_dynamic(
                leading_dim=(1 if d_major == "n" else 0)
            )
            ref_d_tensor.element_type = d_dtype
            ref_d_tensor = cutlass_torch.convert_cute_tensor(
                ref,
                ref_d_tensor,
                d_dtype,
                is_dynamic_layout=True,
            )

            ref_d = f8_torch_tensor.cpu()
        else:
            ref_d = ref.to(cutlass_torch.dtype(d_dtype))

        # Reference checking ref_d and gpu_d
        torch.testing.assert_close(gpu_d, ref_d, atol=tolerance, rtol=1e-05)

    from triton.testing import do_bench

    current_stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)

    flops = 2 * m * n * k * l

    repeats = iterations
    warmup = warmup_iterations

    import time

    time.sleep(0.5)
    if ab_dtype.width == 8:
        assert l == 1
        scale_ab = torch.ones((1,), dtype=torch.float32, device="cuda")
        fn_cublas = lambda: torch._scaled_mm(
            a_torch[:, :, 0],
            b_torch[:, :, 0].mT,
            scale_a=scale_ab,
            scale_b=scale_ab,
            out_dtype=torch.bfloat16,
            # use_fast_accum=fp8_fast_accum,
        )
    else:
        if c_torch is None:
            fn_cublas = lambda: torch.matmul(a_torch.permute(2, 0, 1), b_torch.permute(2, 0, 1).mT)
        else:
            c_torch_convert = c_torch.to(a_torch.dtype)  # In case C is in FP32
            fn_cublas = lambda: torch.baddbmm(
                c_torch_convert.permute(2, 0, 1),
                a_torch.permute(2, 0, 1),
                b_torch.permute(2, 0, 1).mT,
            )
    timing_cublas = do_bench(fn_cublas, warmup=warmup, rep=repeats)
    tflops_cublas = flops / (timing_cublas * 1e9)  # Convert to TFlops
    print(f"CuBLAS Average time: {timing_cublas:.3f} ms, TFLOPS: {tflops_cublas:.1f}")

    time.sleep(0.5)
    fn = lambda: compiled_gemm(
        mA, mB, mD, mC, epi_args, scheduler_args, varlen_args, current_stream
    )
    timing = do_bench(fn, warmup=warmup, rep=repeats)
    tflops = flops / (timing * 1e9)  # Convert to TFlops
    print(f"Cute-DSL Average time: {timing:.3f} ms, TFLOPS: {tflops:.1f}")

    # time.sleep(0.5)
    # timing_cublas = do_bench(fn_cublas, warmup=warmup, rep=repeats)
    # tflops_cublas = flops / (timing_cublas * 1e9)  # Convert to TFlops
    # print(f"CuBLAS Average time: {timing_cublas:.3f} ms, TFLOPS: {tflops_cublas:.1f}")


if __name__ == "__main__":

    def parse_comma_separated_ints(s: str) -> Tuple[int, ...]:
        try:
            return tuple(int(x.strip()) for x in s.split(","))
        except ValueError:
            raise argparse.ArgumentTypeError("Invalid format. Expected comma-separated integers.")

    parser = argparse.ArgumentParser(description="Example of Dense Persistent GEMM on Blackwell.")

    parser.add_argument(
        "--mnkl",
        type=parse_comma_separated_ints,
        default=(256, 256, 512, 1),
        # default=(4096, 4096, 4096, 1),
        help="mnkl dimensions (comma-separated)",
    )
    parser.add_argument(
        "--mma_tiler_mn",
        type=parse_comma_separated_ints,
        default=(128, 128),
        help="Mma tile shape (comma-separated)",
    )
    parser.add_argument(
        "--cluster_shape_mn",
        type=parse_comma_separated_ints,
        default=(1, 1),
        help="Cluster shape (comma-separated)",
    )
    parser.add_argument("--ab_dtype", type=cutlass.dtype, default=cutlass.BFloat16)
    parser.add_argument("--d_dtype", type=cutlass.dtype, default=cutlass.BFloat16)
    parser.add_argument("--c_dtype", type=cutlass.dtype, default=None)
    parser.add_argument("--acc_dtype", type=cutlass.dtype, default=Float32)
    parser.add_argument("--a_major", choices=["k", "m"], type=str, default="k")
    parser.add_argument("--b_major", choices=["k", "n"], type=str, default="k")
    parser.add_argument("--d_major", choices=["n", "m"], type=str, default="n")
    parser.add_argument("--c_major", choices=["n", "m"], type=str, default="n")

    parser.add_argument("--tolerance", type=float, default=3e-02, help="Tolerance for validation")
    parser.add_argument("--warmup_iterations", type=int, default=5, help="Warmup iterations")
    parser.add_argument(
        "--iterations",
        type=int,
        default=30,
        help="Number of iterations to run the kernel",
    )
    parser.add_argument("--skip_ref_check", action="store_true", help="Skip reference checking")
    parser.add_argument(
        "--dynamic_persistent", action="store_true", help="Dynamic persistent kernel"
    )

    args = parser.parse_args()

    if len(args.mnkl) != 4:
        parser.error("--mnkl must contain exactly 4 values")

    if len(args.mma_tiler_mn) != 2:
        parser.error("--mma_tiler_mn must contain exactly 2 values")

    if len(args.cluster_shape_mn) != 2:
        parser.error("--cluster_shape_mn must contain exactly 2 values")

    run(
        args.mnkl,
        args.ab_dtype,
        args.d_dtype,
        args.c_dtype,
        args.acc_dtype,
        args.a_major,
        args.b_major,
        args.d_major,
        args.c_major,
        args.mma_tiler_mn,
        args.cluster_shape_mn,
        args.tolerance,
        args.warmup_iterations,
        args.iterations,
        args.skip_ref_check,
        args.dynamic_persistent,
    )
    print("PASS")
