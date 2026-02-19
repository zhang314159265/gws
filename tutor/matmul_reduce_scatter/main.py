"""
Fuse matmul + reduce-scatter and do benchmarking following Kraken repo.
"""

import argparse
from tabulate import tabulate
from typing import Callable, Any
from contextlib import nullcontext
import functools
import sys
from dataclasses import asdict, dataclass
from collections import defaultdict
import csv
import os

from cuda.bindings import driver

import torch
import torch.distributed as dist
import torch.distributed._symmetric_memory as symm_mem
from triton.tools.tensor_descriptor import TensorDescriptor

import triton
import triton.language as tl

from ..all_reduce_norm.sync import _get_flat_tid, symm_mem_sync, _send_signal
from ..all_reduce_norm.util import do_benchmark_with_event

def shape_input_type(s):
    try:
        M, N, K = map(int, s.split(","))
        return M, N, K
    except Exception as e:
        raise argparse.ArgumentTypeError("Shape must be M, N, K") from e

@dataclass(frozen=True)
class ExperimentConfig:
    shape: tuple[int, int, int]
    dtype: torch.dtype
    backends: list[str]
    baseline_backend: str
    device: torch.device

    def asdict(self):
        # Convert the dataclass instance to a dictionary
        d = asdict(self)
        d.pop("backends", None)
        d.pop("baseline_backend", None)
        d.pop("device", None)
        return d

@dataclass(frozen=True)
class Experiment:
    config: ExperimentConfig
    results: dict[str, float]  # backend -> time in us

    def asdict(self):
        dict1 = self.config.asdict()
        dict2 = self.results
        return {**dict1, **dict2}

def generate_experiment_configs(
    dtype: torch.dtype,
    shapes: list[tuple[int, int, int]],
    backends: list[str],
    device: torch.device,
) -> list[ExperimentConfig]:
    all_configs = []
    for shape in shapes:
        all_configs.append(
            ExperimentConfig(
                shape=shape,
                dtype=dtype,
                backends=backends,
                baseline_backend=backends[0],
                device=device,
            )
        )

    return all_configs

def main():
    # Set up the argument parser
    parser = argparse.ArgumentParser(
        description="Run sweep over sizes for matmul-reduce-scatter. "
    )

    parser.add_argument(
        "--backend",
        type=str,
        nargs="+",
        choices=[
            "nccl",
            "torch_symm_mem",
            "triton",
            "triton_ce",
        ],
        default=["nccl", "torch_symm_mem", "triton",],
        help="Backend to use for Matmul Reduce Scatter. Use first backend as baseline. 'triton' uses gemm_reduce_scatter, 'triton_ce' uses gemm_reduce_scatter_ce_persistent.",
    )

    parser.add_argument(
        "--shape",
        type=shape_input_type,
        nargs="+",
        default=[
            (m, 6656, k)
            for m in [2**x for x in range(7, 11)]
            for k in [2**x for x in range(12, 16)]
        ],
        help="matmul shapes: M, N, K. (M, K) @ (K, N) -> (M, N)",
    )

    parser.add_argument("-dtype", type=str, help="dtype", default="float32")

    args = parser.parse_args()
    args.dtype = getattr(torch, args.dtype)

    if "LOCAL_RANK" not in os.environ:
        print(
            "Error: LOCAL_RANK environment variable is not defined. Are you running with torchrun? "
        )
        sys.exit(1)

    try:
        local_rank = int(os.environ["LOCAL_RANK"])
    except ValueError:
        print(
            "Error: LOCAL_RANK environment variable must be a valid integer. Are you running with torchrun? "
        )
        sys.exit(1)

    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)
    dist.init_process_group("nccl")
    torch.manual_seed(42 + local_rank)

    results = []
    configs = generate_experiment_configs(args.dtype, args.shape, args.backend, device)
    for config in configs:
        results.append(
            Experiment(
                config,
                run_experiment(config),
            )
        )
    if dist.get_rank() == 0:
        print_results(results)
    dist.destroy_process_group()

def print_results(results: list[Experiment]):
    table_data = defaultdict(list)

    for experiment in results:
        baseline_time = experiment.results[experiment.config.baseline_backend]
        min_time = float("inf")
        best_backend = experiment.config.baseline_backend
        backends = experiment.config.backends
        for key, value in experiment.asdict().items():
            if key in backends:
                if value < min_time:
                    min_time = value
                    best_backend = key
                table_data[key].append(value)
            else:
                table_data[key].append(value)
        table_data[f"Speedup over {experiment.config.baseline_backend}"].append(
            baseline_time / min_time
        )
        table_data["Best Backend"].append(best_backend)
    print(tabulate(table_data, headers="keys", tablefmt="github", floatfmt=".3f"))

def nccl_mem_gemm_rs(a, b):
    from torch.distributed._functional_collectives import (
        reduce_scatter_tensor,
        wait_tensor,
    )

    gemm_output = torch.matmul(a, b)
    rs_o = reduce_scatter_tensor(
        gemm_output, "sum", scatter_dim=0, group=dist.group.WORLD
    )
    return wait_tensor(rs_o)

def torch_symm_mem_gemm_rs(a, b):
    return torch.ops.symm_mem.fused_matmul_reduce_scatter(
        a, b, "sum", scatter_dim=0, group_name=dist.group.WORLD.group_name
    )

def get_single_backend_fn(backend: str):
    if backend == "nccl":
        return nccl_mem_gemm_rs
    if backend == "torch_symm_mem":
        return torch_symm_mem_gemm_rs
    if backend == "triton":
        return fused_gemm_reduce_scatter
    if backend == "triton_ce":
        return gemm_reduce_scatter_ce_persistent
    raise NotImplementedError(backend)

def clone_symm_mem_tensor(tensor: torch.Tensor) -> torch.Tensor:
    symm_mem_tensor = symm_mem.empty(
        tensor.shape,
        dtype=tensor.dtype,
        device=tensor.device,
    )
    symm_mem.rendezvous(symm_mem_tensor, dist.group.WORLD.group_name)
    symm_mem_tensor.copy_(tensor)
    return symm_mem_tensor

def check_numeric(ref, act, tol):
    dif = (ref - act).abs()
    thres = act.abs() * tol + tol
    cnt = (dif > thres).sum()
    if cnt / ref.numel() > 0.01:
        torch.testing.assert_close(ref, act, atol=tol, rtol=tol)

def run_experiment(config: ExperimentConfig, tol=2.0) -> dict[str, float]:
    M, N, K = config.shape
    a = symm_mem.empty(
        (M, K),
        dtype=config.dtype,
        device=config.device,
    ).normal_()
    b = torch.randn((K, N), device=config.device, dtype=config.dtype).T.contiguous().T
    symm_mem.rendezvous(a, dist.group.WORLD.group_name)

    input_tensors = {backend: clone_symm_mem_tensor(a) for backend in config.backends}
    golden_inp = clone_symm_mem_tensor(a)

    golden_o = get_single_backend_fn(config.baseline_backend)(golden_inp, b)

    results = {}
    for backend in config.backends:
        fn = get_single_backend_fn(backend)
        inp = input_tensors[backend]

        test_o = fn(inp, b)
        check_numeric(golden_o[0], test_o[0], tol)

        target_fn = functools.partial(fn, inp, b)
        results[backend] = do_benchmark_with_event(target_fn, ref=None, flush_l2=True)

    return results

def fused_gemm_reduce_scatter(a: torch.Tensor, b: torch.Tensor, **kwargs) -> torch.Tensor:
    """
    Fused GEMM + Reduce-Scatter operation.
    Computes C = A @ B on each rank, then performs reduce-scatter to sum results
    and scatter them along the M dimension.
    Args:
        a: Input matrix A of shape (M, K)
        b: Input matrix B of shape (K, N)
    Returns:
        Output matrix of shape (M / world_size, N) containing the reduce-scattered result.
    """

    assert a.shape[1] == b.shape[0], (
        "Inner dimensions must match for matrix multiplication"
    )
    M, K = a.shape
    _, N = b.shape

    group = kwargs.get("group")
    group = dist.group.WORLD if group is None else group
    world_size = dist.get_world_size(group)
    rank = dist.get_rank(group)

    assert M % world_size == 0, (
        f"M dimension ({M}) must be divisible by world_size ({world_size})"
    )

    # Configuration
    BLOCK_SIZE_M = kwargs.get("BLOCK_SIZE_M", 64)
    BLOCK_SIZE_N = kwargs.get("BLOCK_SIZE_N", 64)
    BLOCK_SIZE_K = kwargs.get("BLOCK_SIZE_K", 32)
    GROUP_SIZE_M = kwargs.get("GROUP_SIZE_M", 8)
    num_warps = kwargs.get("num_warps", 4)
    num_stages = kwargs.get("num_stages", 3)
    assert a.dtype == b.dtype, "Input tensors must have the same dtype"

    M_scatter = M // world_size
    # Create output tensor for the scattered result
    output = torch.empty((M_scatter, N), dtype=a.dtype, device=a.device)

    symm_mem_hdl = symm_mem.get_symm_mem_workspace(
        group.group_name, min_size=M * N * a.element_size()
    )

    buf_list = [
        symm_mem_hdl.get_buffer(i, [M, N], a.dtype, 0)
        for i in range(symm_mem_hdl.world_size)
    ]
    buf_tuple = tuple(buf_list)
    gemm_buffer = buf_list[rank]

    # Launch kernel
    def grid(META):
        return (
            triton.cdiv(M, META["BLOCK_SIZE_M"]) * triton.cdiv(N, META["BLOCK_SIZE_N"]),
        )

    gemm_reduce_scatter_kernel[grid](
        a,
        b,
        buf_tuple,
        symm_mem_hdl.signal_pad_ptrs_dev,
        output,
        M=M,
        N=N,
        K=K,
        stride_am=a.stride(0),
        stride_ak=a.stride(1),
        stride_bk=b.stride(0),
        stride_bn=b.stride(1),
        stride_symm_m=gemm_buffer.stride(0),
        stride_symm_n=gemm_buffer.stride(1),
        stride_out_m=output.stride(0),
        stride_out_n=output.stride(1),
        rank=rank,
        world_size=world_size,
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        BLOCK_SIZE_K=BLOCK_SIZE_K,
        GROUP_SIZE_M=GROUP_SIZE_M,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    return output

@triton.jit
def gemm_reduce_scatter_kernel(
    a_ptr,
    b_ptr,
    buf_tuple,
    signal_pad_ptrs,
    output_ptr,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    stride_am: tl.constexpr,
    stride_ak: tl.constexpr,
    stride_bk: tl.constexpr,
    stride_bn: tl.constexpr,
    stride_symm_m: tl.constexpr,
    stride_symm_n: tl.constexpr,
    stride_out_m: tl.constexpr,
    stride_out_n: tl.constexpr,
    rank: tl.constexpr,
    world_size: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
):
    """
    Fused GEMM + Reduce-Scatter kernel.
    Computes C = A @ B locally, then performs reduce-scatter across ranks.
    The result is scattered along the M dimension.
    """

    # 1. Program ID and Tiling Calculation
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    # 2. Local GEMM computation
    # We do A @ B and C gets stored in rank's symm mem buffer

    # Create pointers for the first blocks of A and B
    offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

    # GEMM Computation
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(tl.cdiv(K, BLOCK_SIZE_K)):
        a = tl.load(a_ptrs, mask=(offs_k[None, :] < K - k * BLOCK_SIZE_K), other=0.0)
        b = tl.load(b_ptrs, mask=(offs_k[:, None] < K - k * BLOCK_SIZE_K), other=0.0)
        accumulator = tl.dot(a, b, accumulator)
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk

    # This is the full C matrix
    c_local = accumulator.to(a_ptr.dtype.element_ty)

    # Get this rank's buffer in the symmetric memory space
    my_buffer_ptr = buf_tuple[rank]

    # Calculate where to store this tile in the buffer
    offs_symm_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_symm_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)

    # Create pointers to the symmetric memory location
    symm_ptrs = (
        my_buffer_ptr
        + stride_symm_m * offs_symm_m[:, None]
        + stride_symm_n * offs_symm_n[None, :]
    )

    # Store the C in the rank's symmetric memory buffer
    mask_mn = (offs_symm_m[:, None] < M) & (offs_symm_n[None, :] < N)
    tl.store(symm_ptrs, c_local, mask=mask_mn)

    # synchronize
    symm_mem_sync(
        signal_pad_ptrs, None, rank, world_size, hasSubsequentMemAccess=True
    )

    # Reduce Scatter logic: For each tile in the rank's assigned row slice (along M),
    # sum corresponding tiles from all ranks' buffers and store the reduced tile directly in the local output.
    # This is to avoid full global sum on any rank.

    # Compute the size of each scattered slice (rows per rank)
    M_scatter = M // world_size

    # Figure out rank's assigned row range in the global output
    my_scatter_start_row = rank * M_scatter
    my_scatter_end_row = (rank + 1) * M_scatter

    # Get the starting row of the current tile this block is handling
    current_tile_start_row = pid_m * BLOCK_SIZE_M

    # If the program's tile falls into scattered output slice for this rank
    if (current_tile_start_row >= my_scatter_start_row) and (
        current_tile_start_row < my_scatter_end_row
    ):
        # This block is responsible for computing a tile of the final output

        # Reduce results from all ranks for this specific tile
        acc_reduce = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
        for i in tl.static_range(world_size):
            buffer_rank_ptr = buf_tuple[i]
            remote_tile_ptrs = (
                buffer_rank_ptr
                + stride_symm_m * offs_symm_m[:, None]
                + stride_symm_n * offs_symm_n[None, :]
            )
            c_block = tl.load(remote_tile_ptrs, mask=mask_mn, other=0.0)
            acc_reduce += c_block

        # Calculate offset into the local scattered output tensor
        offs_out_m = (
            current_tile_start_row - my_scatter_start_row + tl.arange(0, BLOCK_SIZE_M)
        )

        offs_out_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)

        output_ptrs = (
            output_ptr
            + stride_out_m * offs_out_m[:, None]
            + stride_out_n * offs_out_n[None, :]
        )

        mask_out = (offs_out_m[:, None] < M_scatter) & (offs_out_n[None, :] < N)

        # Store the reduced tile to the output and cast to orig dtype
        tl.store(output_ptrs, acc_reduce.to(output_ptr.dtype.element_ty), mask=mask_out)

    symm_mem_sync(
        signal_pad_ptrs, None, rank, world_size, hasPreviousMemAccess=True
    )


# TODO ============== SPLIT ==============

# Command: torchrun --nproc-per-node=8 -m tutor.matmul_reduce_scatter.main --shape 4096,4096,4096 --backend nccl torch_symm_mem triton triton_ce

def _matmul_launch_metadata(grid, kernel, args):
    ret = {}
    M, N, K = args["M"], args["N"], args["K"]
    ret["name"] = f"{kernel.name} [M={M}, N={N}, K={K}]"
    if "c_desc" in args:
        bytes_per_elem = args["c_desc"].base.element_size()
    else:
        bytes_per_elem = 1 if args["FP8_OUTPUT"] else 2
    ret[f"flops{bytes_per_elem * 8}"] = 2.0 * M * N * K
    ret["bytes"] = bytes_per_elem * (M * K + N * K)
    return ret


_tma_desc_cache = {}


def _create_2d_tma_descriptor(tensor: torch.Tensor, block_dim1: int, block_dim0: int) -> TensorDescriptor:
    global _tma_desc_cache
    block_shape = [int(block_dim1), int(block_dim0)]
    key = (
        int(tensor.data_ptr()),
        tuple(int(size) for size in tensor.shape),
        tuple(int(stride) for stride in tensor.stride()),
        tuple(block_shape),
        tensor.dtype,
        tensor.device,
    )
    desc = _tma_desc_cache.get(key)
    if desc is None:
        desc = TensorDescriptor.from_tensor(tensor, block_shape)
        _tma_desc_cache[key] = desc
    return desc


@triton.jit(launch_metadata=_matmul_launch_metadata)
def _gemm_producer_persistent_kernel(
    a_desc,
    b_desc,
    c_desc,
    progress_ptr,
    signal_pad_ptr,
    M,
    N,
    K,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    RANK: tl.constexpr,
    WORLD_SIZE: tl.constexpr,
    FP8_OUTPUT: tl.constexpr,
    NUM_SMS: tl.constexpr,
):
    dtype = tl.float8e4nv if FP8_OUTPUT else tl.bfloat16
    start_pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    k_tiles = tl.cdiv(K, BLOCK_SIZE_K)
    num_tiles = num_pid_m * num_pid_n

    tiles_per_SM = num_tiles // NUM_SMS
    if start_pid < num_tiles % NUM_SMS:
        tiles_per_SM += 1

    tile_id = start_pid - NUM_SMS
    ki = -1

    pid_m = 0
    pid_n = 0
    offs_am = 0
    offs_bn = 0

    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    M_per_rank = M // WORLD_SIZE

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    pid_m_offset = (RANK + 1) * M_per_rank // BLOCK_SIZE_M

    for _ in range(k_tiles * tiles_per_SM):
        ki = tl.where(ki == k_tiles - 1, 0, ki + 1)
        if ki == 0:
            tile_id += NUM_SMS
            group_id = tile_id // num_pid_in_group
            first_pid_m = group_id * GROUP_SIZE_M
            group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
            pid_m = first_pid_m + (tile_id % group_size_m)
            pid_n = (tile_id % num_pid_in_group) // group_size_m

            # Pivot tile_id so that M tiles are processed in communication order.
            # This pivot preserves the prior swizzling.
            pid_m = (pid_m + pid_m_offset) % num_pid_m

            offs_am = pid_m * BLOCK_SIZE_M
            offs_bn = pid_n * BLOCK_SIZE_N

        offs_k = ki * BLOCK_SIZE_K

        a = a_desc.load([offs_am, offs_k])
        b = b_desc.load([offs_bn, offs_k])
        accumulator = tl.dot(a, b.T, accumulator)

        if ki == k_tiles - 1:
            c = accumulator.to(dtype)
            c_desc.store([offs_am, offs_bn], c)

            # calculate progress and send signals to corresponding ranks
            scatter_start = offs_am // M_per_rank
            scatter_end = (offs_am + BLOCK_SIZE_M - 1) // M_per_rank
            scatter_end = min(scatter_end, WORLD_SIZE - 1)

            for rank in range(scatter_start, scatter_end + 1):
                m_start = M_per_rank * rank
                m_end = M_per_rank * (rank + 1) - 1
                tiled_m_start = m_start // BLOCK_SIZE_M
                tiled_m_end = m_end // BLOCK_SIZE_M
                tiled_m_size = tiled_m_end - tiled_m_start + 1
                val = tl.atomic_add(progress_ptr + rank, 1, sem="release", scope="gpu")
                if val == tiled_m_size * num_pid_n - 1:
                    send_addr = signal_pad_ptr + rank
                    if _get_flat_tid() == 0:
                        _send_signal(send_addr, "release")

            accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)


def copy_engine_scatter(
    inp: torch.Tensor,
    output: torch.Tensor,  # Must be symmetric tensor
    signal_pad: torch.Tensor,
    group: dist.ProcessGroup | None = None,
):
    assert output.is_contiguous()
    M, N = inp.shape

    symm_mem_hdl = symm_mem.get_symm_mem_workspace(
        group.group_name, min_size=output.numel() * output.element_size()
    )

    rank = symm_mem_hdl.rank
    world_size = symm_mem_hdl.world_size
    M_per_rank = M // world_size

    # copy gemm tiles to corresponding ranks
    stream = torch.cuda.current_stream()
    for step in range(world_size):
        remote_rank = (rank + step + 1) % world_size

        # wait signal from gemm kernel
        signal_pad_ptr = signal_pad.data_ptr()
        signal_ele_size = signal_pad.element_size()
        wait_addr = signal_pad_ptr + signal_ele_size * remote_rank
        driver.cuStreamWaitValue32(
            stream.cuda_stream,
            wait_addr,
            1,
            driver.CUstreamWaitValue_flags.CU_STREAM_WAIT_VALUE_EQ,
        )

        offset = rank * M_per_rank * N
        remote_buf = symm_mem_hdl.get_buffer(
            remote_rank, [M_per_rank, N], inp.dtype, offset
        )
        src_buf = inp[remote_rank * M_per_rank : (remote_rank + 1) * M_per_rank, :]
        remote_buf.copy_(src_buf)


def gemm_producer_w_progress(
    a: torch.Tensor,
    b: torch.Tensor,
    gemm_out: torch.Tensor,
    progress: torch.Tensor,
    signal_pad: torch.Tensor,
    configs: dict,
    group: dist.ProcessGroup | None = None,
):
    M, K = a.shape
    Kb, N = b.shape
    assert Kb == K, "Inner dimensions must match for matrix multiplication"
    assert a.dtype == b.dtype, "Input dtypes must match"

    bT = b.T.contiguous()

    desc_a = _create_2d_tma_descriptor(
        a,
        configs["BLOCK_SIZE_M"],
        configs["BLOCK_SIZE_K"],
    )
    desc_bt = _create_2d_tma_descriptor(
        bT,
        configs["BLOCK_SIZE_N"],
        configs["BLOCK_SIZE_K"],
    )
    desc_c = _create_2d_tma_descriptor(
        gemm_out,
        configs["BLOCK_SIZE_M"],
        configs["BLOCK_SIZE_N"],
    )

    configs["NUM_SMS"] = torch.cuda.get_device_properties(
        a.device
    ).multi_processor_count

    grid = lambda META: (  # noqa: E731
        min(
            configs["NUM_SMS"],
            triton.cdiv(M, META["BLOCK_SIZE_M"]) * triton.cdiv(N, META["BLOCK_SIZE_N"]),
        ),
    )

    group = dist.group.WORLD if group is None else group

    _gemm_producer_persistent_kernel[grid](
        desc_a,
        desc_bt,
        desc_c,
        progress,
        signal_pad,
        M,
        N,
        K,
        BLOCK_SIZE_M=configs["BLOCK_SIZE_M"],
        BLOCK_SIZE_N=configs["BLOCK_SIZE_N"],
        BLOCK_SIZE_K=configs["BLOCK_SIZE_K"],
        GROUP_SIZE_M=configs["GROUP_SIZE_M"],
        RANK=configs["RANK"],
        WORLD_SIZE=configs["WORLD_SIZE"],
        FP8_OUTPUT=a.dtype == torch.float8_e4m3fn,
        NUM_SMS=configs["NUM_SMS"],
        num_stages=configs["num_stages"],
        num_warps=configs["num_warps"],
    )


@triton.jit
def _reduce_persistent_kernel(
    in_desc,  # [M, N]
    out_desc,  # [M_per_rank, N]
    M_per_rank,
    N,
    RANK: tl.constexpr,
    WORLD_SIZE: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr = 256,
    BLOCK_SIZE_N: tl.constexpr = 64,
):
    pid = tl.program_id(axis=0)
    num_pid = tl.num_programs(axis=0)
    num_tiles_m = tl.cdiv(M_per_rank, BLOCK_SIZE_M)
    num_tiles_n = tl.cdiv(N, BLOCK_SIZE_N)
    total_tiles = num_tiles_m * num_tiles_n
    for tile_id in range(pid, total_tiles, num_pid):
        tile_id_m = tile_id // num_tiles_n
        tile_id_n = tile_id % num_tiles_n
        cur_rank = (RANK + 1) % WORLD_SIZE
        accum = in_desc.load(
            [
                tile_id_m * BLOCK_SIZE_M + cur_rank * M_per_rank,
                tile_id_n * BLOCK_SIZE_N,
            ]
        )
        for i in range(1, WORLD_SIZE):
            cur_rank = (i + RANK + 1) % WORLD_SIZE
            data = in_desc.load(
                [
                    tile_id_m * BLOCK_SIZE_M + cur_rank * M_per_rank,
                    tile_id_n * BLOCK_SIZE_N,
                ]
            )
            accum += data

        out_desc.store([tile_id_m * BLOCK_SIZE_M, tile_id_n * BLOCK_SIZE_N], accum)


def reduce(
    inp: torch.Tensor,  # scatter_out with shape [M, N]
    output: torch.Tensor,  # [M_per_rank, N]
    configs: dict,
):
    M, N = inp.shape
    M_per_rank = M // configs["WORLD_SIZE"]
    assert output.shape[0] == M_per_rank and M % configs["WORLD_SIZE"] == 0

    BLOCK_SIZE_M = 256
    BLOCK_SIZE_N = 64

    in_desc = _create_2d_tma_descriptor(
        inp,
        BLOCK_SIZE_M,
        BLOCK_SIZE_N,
    )
    out_desc = _create_2d_tma_descriptor(
        output,
        BLOCK_SIZE_M,
        BLOCK_SIZE_N,
    )

    grid = lambda META: (  # noqa: E731
        triton.cdiv(M_per_rank, META["BLOCK_SIZE_M"])
        * triton.cdiv(N, META["BLOCK_SIZE_N"]),
    )
    _reduce_persistent_kernel[grid](
        in_desc,
        out_desc,
        M_per_rank,
        N,
        RANK=configs["RANK"],
        WORLD_SIZE=configs["WORLD_SIZE"],
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        num_warps=4,
    )

    return output


def gemm_reduce_scatter_ce_persistent(
    a: torch.Tensor,
    b: torch.Tensor,
    reduce_op: str = "sum",  # only support sum for now
    group: dist.ProcessGroup | None = None,
    **kwargs,
) -> tuple[torch.Tensor, torch.Tensor]:
    M = a.shape[0]
    N = b.shape[1]

    group = dist.group.WORLD if group is None else group
    gemm_out = torch.empty((M, N), dtype=a.dtype, device=a.device)
    symm_mem_hdl = symm_mem.get_symm_mem_workspace(
        group.group_name, min_size=M * N * a.element_size()
    )
    scatter_out = symm_mem_hdl.get_buffer(symm_mem_hdl.rank, [M, N], a.dtype, 0)
    world_size = symm_mem_hdl.world_size

    assert M % world_size == 0
    M_per_rank = M // world_size
    backend_stream = symm_mem._get_backend_stream(priority=-1)
    backend_stream.wait_stream(torch.cuda.current_stream())

    output = torch.empty((M_per_rank, N), dtype=a.dtype, device=a.device)

    configs = {
        "BLOCK_SIZE_M": kwargs.get("block_size_m", 128),
        "BLOCK_SIZE_N": kwargs.get("block_size_n", 128),
        "BLOCK_SIZE_K": kwargs.get("block_size_k", 64),
        "GROUP_SIZE_M": kwargs.get("group_size_m", 8),
        "num_stages": kwargs.get("num_stages", 2),
        "num_warps": kwargs.get("num_warps", 8),
    }
    configs["RANK"] = symm_mem_hdl.rank
    configs["WORLD_SIZE"] = world_size

    progress = torch.zeros(world_size, dtype=torch.uint32, device=a.device)
    signal_pad = torch.zeros(world_size, dtype=torch.uint32, device=a.device)

    gemm_producer_w_progress(a, b, gemm_out, progress, signal_pad, configs)

    with backend_stream:
        copy_engine_scatter(gemm_out, scatter_out, signal_pad, group)

    torch.cuda.current_stream().wait_stream(backend_stream)
    symm_mem_hdl.barrier()

    reduce(scatter_out, output, configs)

    return output

if __name__ == "__main__":
    main()
