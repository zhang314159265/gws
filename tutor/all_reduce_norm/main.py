"""
Fuse all-reduce + bias + rms-norm and do benchmarking following Kraken repo.
"""

from typing import Callable, Any
from contextlib import nullcontext
import torch.distributed as dist
import torch

import triton
import triton.language as tl
from triton.language.math import rsqrt as tl_rsqrt

import functools
import gc
import os

import torch.distributed._symmetric_memory as symm_mem
import math
from .sync import *

def nccl_all_reduce_bias_rms_norm(x, bias, w):
    y = x.clone()
    dist.all_reduce(y)
    y = y + bias
    return fused_rms_norm(y, w)

@triton.jit
def _rms_norm_kernel(
    x_ptr,
    y_ptr,
    w_ptr,
    eps: tl.constexpr,
    D: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    bt_stride: tl.constexpr,
):
    row_idx = tl.program_id(axis=0).to(tl.int64)
    bt_idx = row_idx
    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < D

    offset = bt_idx * bt_stride
    row_start_ptr = x_ptr + offset
    out_start_ptr = y_ptr + offset
    input_ptrs = row_start_ptr + col_offsets
    output_ptrs = out_start_ptr + col_offsets

    row = tl.load(input_ptrs, mask=mask, other=0.0, eviction_policy="evict_last").to(
        tl.float32
    )
    variance = tl.sum(row * row, axis=0) / D
    rstd = tl_rsqrt(variance + eps)

    w = tl.load(w_ptr + col_offsets, mask=mask, eviction_policy="evict_first").to(
        tl.float32
    )
    tl.store(output_ptrs, row * rstd * w, mask=mask)

def fused_rms_norm(
    x: torch.Tensor,
    w: torch.Tensor,
    eps: float = 1.0e-5,
) -> torch.Tensor:
    y = torch.empty_like(x)
    D = x.shape[-1]
    assert w.is_contiguous()
    assert w.shape == (D,)
    assert x.is_contiguous()
    assert y.is_contiguous()
    assert y.shape == x.shape

    num_blocks = math.prod(x.shape[:-1])
    _rms_norm_kernel[(num_blocks,)](
        x,
        y,
        w,
        eps,
        D,
        BLOCK_SIZE=triton.next_power_of_2(D),
        bt_stride=D,
        num_warps=32,
    )
    return y

def do_benchmark_with_event(
    target_fn: Callable[..., Any],
    ref: torch.Tensor,
    warmup_iters: int = 200,
    benchmark_iters: int = 200,
    flush_l2: bool = True,
    tol: float = 0.05,
    profile_ranks: list[int] | None = None,
) -> float:
    act = target_fn()
    torch.testing.assert_close(ref, act, atol=tol, rtol=tol)

    rank = dist.get_rank()
    profile_ranks = profile_ranks or [0]

    if flush_l2:
        cache = torch.empty(int(256e6 // 4), dtype=torch.int, device="cuda")

    for _ in range(warmup_iters):
        target_fn()

    dist.barrier()
    torch.cuda.synchronize()

    begin_events = [
        torch.cuda.Event(enable_timing=True) for _ in range(benchmark_iters)
    ]
    end_events = [torch.cuda.Event(enable_timing=True) for _ in range(benchmark_iters)]

    # if rank in profile_ranks:
    if False:
        from trace_handler import trace_handler

        prof = torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ],
            on_trace_ready=trace_handler,
        )
    else:
        prof = nullcontext()

    with prof:
        torch.cuda._sleep(int(2e7))
        for i in range(benchmark_iters):
            if flush_l2:
                cache.zero_()
            begin_events[i].record()
            target_fn()
            end_events[i].record()
        torch.cuda.synchronize()

    latencies = [
        b.elapsed_time(e) for b, e in zip(begin_events, end_events)
    ]
    return torch.tensor(latencies).median().item() * 1000  # micro-seconds

# def main(max_b: int = 512 * 32, max_t: int = 1, d_size: int = 5120):
def main(max_b: int = 512 * 32, max_t: int = 1, d_size: int = 4096):
    local_rank = int(os.getenv("LOCAL_RANK"))
    dist.init_process_group(device_id=local_rank)

    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)
    torch.manual_seed(42 + local_rank)

    if dist.get_rank() == 0:
        print("Benchmarking ...")

    comm_bytes = []
    runtime_results = []
    b_sizes = [2**k for k in range(max_b.bit_length())]
    t_sizes = [2**k for k in range(max_t.bit_length())]
    for b in b_sizes:
        for t in t_sizes:
            result = benchmark(device, b, t=t, d_size=d_size)
            total_bytes = b * t * d_size * torch.bfloat16.itemsize
            comm_bytes.append(f"b:{b} t:{t} d:{d_size} bytes:{total_bytes}")

            runtime_results.append([])
            for v in result.values():
                runtime_results[-1].append(v)

    experiments = list(result.keys())
    if dist.get_rank() == 0 and os.environ.get("PLOT") == "1":
        plot_experiment_comparison(
            comm_bytes,
            experiments,
            runtime_results,
            "/tmp/benchmark_all_reduce_bias_rms_norm.png",
        )
    dist.destroy_process_group()

@torch.no_grad()
def benchmark(device: torch.device, b: int, t: int, d_size: int) -> dict[str, float]:
    """
    NOTE: bias and w are the same across all ranks for this workload.

    dist.all_reduce(x)
    y = x + bias
    y = rms_norm(y, w)
    """
    gc.disable()

    all_benchmarks = create_benchmarks(b, t, d_size, device, torch.bfloat16)
    ref = all_benchmarks["nccl_ring"]()
    torch.cuda.synchronize()

    results = {}
    for k, v in all_benchmarks.items():
        runtime = do_benchmark_with_event(v, ref)
        results[k] = runtime

    result_string = "\t".join([f"{k}: {v:.2f} us " for k, v in results.items()])
    if dist.get_rank() == 0:
        print(
            f"b: {b:5} \t"
            # f"t: {t} \t"
            f"d: {d_size} \t"
            # f"bytes: {b * t * d_size * torch.bfloat16.itemsize} \t"
            f"{result_string}"
        )

    return results

def create_benchmarks(b, t, d_size, device, dtype):
    x = torch.randn(b, t, d_size, dtype=dtype, device=device)

    # Ensure that bias and w are the same across ranks
    torch.manual_seed(42)
    w = torch.randn(d_size, dtype=dtype, device=device)
    bias = torch.randn(b, t, d_size, dtype=dtype, device=device)

    all_functions = {
        "nccl_ring": nccl_all_reduce_bias_rms_norm,
        # "one_shot_bias_fused + rms_norm": one_shot_all_reduce_bias_with_rms_norm,
        # "two_shot_bias_fused + rms_norm": two_shot_all_reduce_bias_with_rms_norm,
        # "one_shot_bias_rms_norm_fused": one_shot_all_reduce_bias_rms_norm,
        "two_shot_bias_rms_norm_fused": two_shot_all_reduce_bias_rms_norm,
        "two_shot_bias_rms_norm_fused_split_column": two_shot_all_reduce_bias_rms_norm_split_column,
    }
    all_benchmarks = {}
    for k, v in all_functions.items():
        if k == "nccl_ring":
            all_benchmarks[k] = functools.partial(
                v, x=x.clone(), bias=bias, w=w
            )
        else:
            symm_mem_input = symm_mem.empty(b, t, d_size, dtype=dtype, device=device)
            symm_mem.rendezvous(symm_mem_input, dist.group.WORLD.group_name)
            all_benchmarks[k] = functools.partial(
                v,
                x=x.clone(),
                bias=bias,
                rms_weight=w,
                symm_mem_input=symm_mem_input,
            )

    return all_benchmarks

def one_shot_all_reduce_bias_with_rms_norm(x, bias, rms_weight, symm_mem_input):
    y = torch.empty_like(x)
    fused_one_shot_all_reduce_bias(symm_mem_input, x, bias, y)
    return fused_rms_norm(y, rms_weight)

def fused_one_shot_all_reduce_bias(
    symm_mem_buffer: torch.Tensor,
    input_tensor: torch.Tensor,
    bias: torch.Tensor | None,
    output: torch.Tensor,
    max_num_blocks: int = 24,
    BLOCK_SIZE: int = 4096,
    group: dist.ProcessGroup | None = None,
) -> None:
    """
    One-shot all-reduce operation with optional bias addition on the input.

    output = all_reduce(input)
    output = output + bias if bias is not None else output

    NOTE: that bias is assumed to be the same as this use case is for inference.

    This kernel uses a persistent execution style, launching up to 24 blocks,
    with blocks iterating over the input tensor until all elements are
    processed.

    Args:
        symm_mem_buffer (torch.Tensor): The symmetric memory buffer.
        input_tensor (torch.Tensor): The input tensor to be reduced. Must be of dtype
            torch.bfloat16 and 128-bit aligned.
        bias (torch.Tensor | None): The bias tensor to be added to the reduced
            input. If None, no bias is added.
        output (torch.Tensor): The tensor where the result of the all-reduce
            operation is stored.
        group (dist.ProcessGroup | None, optional): The process group to use for
            the all-reduce operation. If None, the default process group will be
            used.
        max_num_blocks (int, optional): The maximum number of blocks to launch.
        BLOCK_SIZE (int, optional): The BLOCK_SIZE parameter for the kernel.

    Returns:
        None
    """

    group = group or dist.group.WORLD
    symm_mem_hdl = symm_mem.rendezvous(symm_mem_buffer, group=group)
    if symm_mem_hdl is None:
        raise ValueError("symm_mem_buffer much be a valid symmetric memory tensor.")
    num_blocks = min(triton.cdiv(input_tensor.numel(), BLOCK_SIZE), max_num_blocks)

    assert input_tensor.dtype == torch.bfloat16, "Only bfloat16 is supported for now."
    assert input_tensor.numel() % 8 == 0, (
        "The number of elements must be 128-bit aligned."
    )

    num_warps = 32

    kernel = one_shot_all_reduce_bias_kernel[(num_blocks,)](
        symm_mem_hdl.buffer_ptrs_dev,
        symm_mem_hdl.signal_pad_ptrs_dev,
        input_tensor,
        bias,
        output,
        numel=input_tensor.numel(),
        has_bias=bias is not None,
        world_size=symm_mem_hdl.world_size,
        rank=symm_mem_hdl.rank,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=num_warps,
    )

    dump_kernel = os.environ.get("SYMM_DUMP_KERNEL", "0") == "1"
    if dump_kernel and torch.distributed.get_rank() == 0:
        log.log_triton_kernel(kernel)

@triton.jit
def one_shot_all_reduce_bias_kernel(
    symm_mem_buffer_ptrs,
    symm_mem_signal_pad_ptrs,
    input_ptr,
    bias_ptr,
    output_ptr,
    numel,
    has_bias: tl.constexpr,
    rank: tl.constexpr,
    world_size: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """
    One-shot all-reduce operation with optional bias addition on the input.

    Args:
        symm_mem_buffer_ptrs: Pointer to the symmetric memory buffer pointers.
        symm_mem_signal_pad_ptrs: Pointer to the signal pad pointers for synchronization.
        input_ptr: Pointer to the input tensor data.
        bias_ptr: Pointer to the bias tensor data.
        output_ptr: Pointer to the output tensor data.
        numel: The total number of elements in the input tensor to be processed.
        has_bias (tl.constexpr): Flag indicating whether a bias is present.
        rank (tl.constexpr): The rank of the current device in the symm_mem group.
        world_size (tl.constexpr): Total number of devices in the symm_mem group.
        BLOCK_SIZE (tl.constexpr): The size of each block for processing.

    Returns:
        None
    """

    pid = tl.program_id(axis=0)
    input_ptr = tl.multiple_of(input_ptr, 16)
    buffer_ptrs = symm_mem_buffer_ptrs.to(tl.pointer_type(tl.uint64))

    # Copy the input to the symmetric memory buffer.
    buffer_ptr = tl.load(buffer_ptrs + rank).to(tl.pointer_type(tl.bfloat16))
    buffer_ptr = tl.multiple_of(buffer_ptr, 16)
    block_start = pid * BLOCK_SIZE
    while block_start < numel:
        offsets = block_start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < numel
        val = tl.load(input_ptr + offsets, mask=mask)
        tl.store(buffer_ptr + offsets, val, mask=mask)
        block_start += tl.num_programs(axis=0) * BLOCK_SIZE

    symm_mem_sync(
        symm_mem_signal_pad_ptrs,
        None,
        rank,
        world_size,
        hasPreviousMemAccess=True,
        hasSubsequentMemAccess=True,
    )

    block_start = pid * BLOCK_SIZE
    while block_start < numel:
        offsets = block_start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < numel
        if has_bias:
            bias_ptr = tl.multiple_of(bias_ptr, 16)
            acc = tl.load(bias_ptr + offsets, mask=mask).to(tl.float32)
        else:
            acc = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)

        # One-shot all-reduce
        for i in range(world_size):
            buffer_ptr = tl.load(buffer_ptrs + i).to(tl.pointer_type(tl.bfloat16))
            buffer_ptr = tl.multiple_of(buffer_ptr, 16)
            val = tl.load(buffer_ptr + offsets, mask=mask).to(tl.float32)
            acc += val

        tl.store(output_ptr + offsets, acc, mask=mask)
        block_start += tl.num_programs(axis=0) * BLOCK_SIZE

    symm_mem_sync(
        symm_mem_signal_pad_ptrs,
        None,
        rank,
        world_size,
        hasPreviousMemAccess=True,
    )

def one_shot_all_reduce_bias_rms_norm(
    symm_mem_input: torch.Tensor,
    x: torch.Tensor,
    bias: torch.Tensor,
    rms_weight: torch.Tensor,
    eps: float = 1.0e-5,
    BLOCK_SIZE: int = 1024,
    group: dist.ProcessGroup | None = None,
) -> None:
    """This function performs a one_shot all-reduce, bias addition and RMSNorm.

    dist.all_reduce(x)
    x = x + bias
    y = F.rms_norm(x, x.shape[-1], w, eps)

    Args:
        symm_mem_input (torch.Tensor): The symmetric memory buffer.
        x (torch.Tensor): The input tensor to be reduced.
        bias (torch.Tensor): The bias tensor to be added to the reduced input.
        w (torch.Tensor): The weights tensor for RMS normalization.
        y (torch.Tensor): The output tensor to store the result.
        eps (float, optional): The epsilon value for RMSNorm. Default is 1.0e-5.
        BLOCK_SIZE (int, optional): The BLOCK_SIZE parameter for the kernel.
        group (dist.ProcessGroup, optional): The process group for allreduce.
            Default is None which uses the WORLD process group.

    Returns:
        torch.Tensor: The resulting tensor after all-reduce, bias addition, and
        RMS normalization.
    """
    w = rms_weight
    y = torch.empty_like(x)
    D = x.shape[-1]

    assert y.shape == x.shape
    assert y.dtype == x.dtype
    assert y.stride() == x.stride(), (str(x.stride()), str(y.stride()))
    assert w.is_contiguous()
    assert w.shape == (D,)
    assert x.is_contiguous()
    assert y.is_contiguous()

    num_blocks = math.prod(x.shape[:-1])
    num_warps = 32
    group = group or dist.group.WORLD
    symm_mem_hdl = symm_mem.rendezvous(symm_mem_input, group=group)

    # This kernel does not perform well for large B
    # XXX This is not a persistent kernel!
    # XXX should the 512 batch-size cause oob access for the signal pad?
    kernel = one_shot_all_reduce_bias_rms_norm_kernel[(num_blocks,)](
        symm_mem_hdl.buffer_ptrs_dev,
        symm_mem_hdl.signal_pad_ptrs_dev,
        x,
        bias,
        w,
        y,
        eps,
        D=D,
        bt_stride=D,
        rank=symm_mem_hdl.rank,
        world_size=symm_mem_hdl.world_size,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=num_warps,
    )

    dump_kernel = os.environ.get("SYMM_DUMP_KERNEL", "0") == "1"
    if dump_kernel and torch.distributed.get_rank() == 0:
        log.log_triton_kernel(kernel)

    return y

@triton.jit
def one_shot_all_reduce_bias_rms_norm_kernel(
    symm_mem_buffer_ptrs,
    symm_mem_signal_pad_ptrs,
    input_ptr,
    bias_ptr,
    w_ptr,
    y_ptr,
    eps: tl.constexpr,
    D: tl.constexpr,
    bt_stride: tl.constexpr,
    rank: tl.constexpr,
    world_size: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    row_idx = tl.program_id(axis=0).to(tl.int64)
    bt_idx = row_idx
    col_offsets = tl.arange(0, triton.next_power_of_2(D))
    mask = col_offsets < D

    input_ptr = tl.multiple_of(input_ptr, 16)
    bias_ptr = tl.multiple_of(bias_ptr, 16)
    y_ptr = tl.multiple_of(y_ptr, 16)

    offset = bt_idx * bt_stride
    buffer_ptrs = symm_mem_buffer_ptrs.to(tl.pointer_type(tl.uint64))

    # Copy the input, x, to the symmetric memory buffer.
    row = tl.load(input_ptr + offset + col_offsets, mask=mask)
    buffer_ptr = tl.load(buffer_ptrs + rank).to(tl.pointer_type(tl.bfloat16))
    buffer_ptr = tl.multiple_of(buffer_ptr, 16)
    tl.store(buffer_ptr + offset + col_offsets, row, mask=mask)

    symm_mem_sync(
        symm_mem_signal_pad_ptrs,
        None,
        rank,
        world_size,
        hasPreviousMemAccess=True,
        hasSubsequentMemAccess=True,
    )

    # Allreduce + bias
    row = tl.load(
        bias_ptr + offset + col_offsets,
        mask=mask,
    ).to(tl.float32)
    for i in range(world_size):
        buffer_ptr = tl.load(buffer_ptrs + i).to(tl.pointer_type(tl.bfloat16))
        buffer_ptr = tl.multiple_of(buffer_ptr, 16)
        x = tl.load(
            buffer_ptr + offset + col_offsets,
            mask=mask,
        ).to(tl.float32)
        row += x

    # The regular RMSNorm
    variance = tl.sum(row * row, axis=0) / D
    rstd = tl_rsqrt(variance + eps)

    w = tl.load(w_ptr + col_offsets, mask=mask).to(tl.float32)
    tl.store(y_ptr + offset + col_offsets, row * rstd * w, mask=mask)

    symm_mem_sync(
        symm_mem_signal_pad_ptrs,
        None,
        rank,
        world_size,
        hasPreviousMemAccess=True,
    )

@triton.jit
def two_shot_all_reduce_bias_rms_norm_kernel_split_column(
    symm_mem_buffer_ptrs,
    symm_mem_signal_pad_ptrs,
    input_ptr,
    bias_ptr,
    w_ptr,
    y_ptr,
    eps: tl.constexpr,
    D: tl.constexpr,
    bt_stride: tl.constexpr,
    rows_per_block: tl.constexpr,
    rank: tl.constexpr,
    world_size: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """
    This kernel assumes we don't need mask.
    """
    row_idx = tl.program_id(axis=0).to(tl.int64) * rows_per_block
    bt_idx = row_idx
    col_offsets = tl.arange(0, triton.next_power_of_2(D))

    # Each block has to compute the RMSNorm one row per time, and
    # the row size is D.
    mask = col_offsets < D

    col_per_rank: tl.constexpr = D // world_size
    start_col = col_per_rank * rank
    chunk_offsets = tl.arange(0, col_per_rank)

    input_ptr = tl.multiple_of(input_ptr, 16)
    bias_ptr = tl.multiple_of(bias_ptr, 16)
    y_ptr = tl.multiple_of(y_ptr, 16)

    offset = bt_idx * bt_stride
    buffer_ptrs = symm_mem_buffer_ptrs.to(tl.pointer_type(tl.uint64))

    # Copy the input, x, to the symmetric memory buffer.
    buffer_ptr = tl.load(buffer_ptrs + rank).to(tl.pointer_type(tl.bfloat16))
    buffer_ptr = tl.multiple_of(buffer_ptr, 16)
    for i in tl.static_range(rows_per_block):
        row = tl.load(input_ptr + offset + i * D + col_offsets, mask=mask)
        tl.store(buffer_ptr + offset + i * D + col_offsets, row, mask=mask)

    symm_mem_sync(
        symm_mem_signal_pad_ptrs,
        None,
        rank,
        world_size,
        hasPreviousMemAccess=True,
        hasSubsequentMemAccess=True,
    )

    # Two shot allreduce
    local_rank_offsets = (row_idx + tl.arange(0, rows_per_block))[:, None] * bt_stride + (start_col + tl.arange(0, col_per_rank))[None, :]

    acc = tl.load(bias_ptr + local_rank_offsets).to(tl.float32)
    for remote_rank in range(world_size):
        buffer_ptr = tl.load(buffer_ptrs + remote_rank).to(tl.pointer_type(tl.bfloat16))
        buffer_ptr = tl.multiple_of(buffer_ptr, 16)
        val = tl.load(buffer_ptr + local_rank_offsets).to(
            tl.float32
        )
        acc += val

    for remote_rank in range(world_size):
        buffer_ptr = tl.load(buffer_ptrs + remote_rank).to(tl.pointer_type(tl.bfloat16))
        buffer_ptr = tl.multiple_of(buffer_ptr, 16)
        tl.store(buffer_ptr + local_rank_offsets, acc)

    symm_mem_sync(
        symm_mem_signal_pad_ptrs,
        None,
        rank,
        world_size,
        hasPreviousMemAccess=True,
        hasSubsequentMemAccess=True,
    )

    # The regular RMSNorm
    buffer_ptr = tl.load(buffer_ptrs + rank).to(tl.pointer_type(tl.bfloat16))
    buffer_ptr = tl.multiple_of(buffer_ptr, 16)
    for i in tl.static_range(rows_per_block):
        row = tl.load(buffer_ptr + offset + i * D + col_offsets, mask=mask).to(
            tl.float32
        )
        variance = tl.sum(row * row, axis=0) / D
        rstd = tl_rsqrt(variance + eps)

        w = tl.load(w_ptr + col_offsets, mask=mask).to(tl.float32)
        tl.store(y_ptr + offset + i * D + col_offsets, row * rstd * w, mask=mask)

    symm_mem_sync(
        symm_mem_signal_pad_ptrs,
        None,
        rank,
        world_size,
        hasPreviousMemAccess=True,
    )

def two_shot_all_reduce_bias_rms_norm_split_column(
    symm_mem_input: torch.Tensor,
    x: torch.Tensor,
    bias: torch.Tensor,
    rms_weight: torch.Tensor,
    eps: float = 1.0e-5,
    group: dist.ProcessGroup | None = None,
) -> None:
    """
    Split column rather than row for reduce-scatter and all-gather.
    """
    w = rms_weight
    y = torch.empty_like(x)
    D = x.shape[-1]

    assert y.shape == x.shape
    assert y.dtype == x.dtype
    assert y.stride() == x.stride(), (str(x.stride()), str(y.stride()))
    assert w.is_contiguous()
    assert w.shape == (D,)
    assert x.is_contiguous()
    assert y.is_contiguous()

    total_rows = math.prod(x.shape[:-1])

    # We only support certain total_rows to just demonstrate the idea.
    BLOCK_SIZE = 1024
    if total_rows < 2:
        rows_per_block = 1
    elif total_rows <= 32:
        rows_per_block = 2
    elif total_rows <= 64:
        rows_per_block = 4
    else:
        rows_per_block = 8

    num_blocks = total_rows // rows_per_block

    num_warps = 32
    group = group or dist.group.WORLD
    symm_mem_hdl = symm_mem.rendezvous(symm_mem_input, group=group)
    world_size = symm_mem_hdl.world_size
    rank = symm_mem_hdl.rank

    assert D % world_size == 0

    kernel = two_shot_all_reduce_bias_rms_norm_kernel_split_column[(num_blocks,)](
        symm_mem_hdl.buffer_ptrs_dev,
        symm_mem_hdl.signal_pad_ptrs_dev,
        x,
        bias,
        w,
        y,
        eps,
        D=D,
        bt_stride=D,
        rows_per_block=rows_per_block,
        rank=rank,
        world_size=world_size,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=num_warps,
    )

    return y


@triton.jit
def two_shot_all_reduce_bias_rms_norm_kernel(
    symm_mem_buffer_ptrs,
    symm_mem_signal_pad_ptrs,
    input_ptr,
    bias_ptr,
    w_ptr,
    y_ptr,
    eps: tl.constexpr,
    D: tl.constexpr,
    bt_stride: tl.constexpr,
    size_per_rank: tl.constexpr,
    rows_per_block: tl.constexpr,
    rank: tl.constexpr,
    world_size: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    row_idx = tl.program_id(axis=0).to(tl.int64) * rows_per_block
    bt_idx = row_idx
    col_offsets = tl.arange(0, triton.next_power_of_2(D))

    # Each block has to compute the RMSNorm one row per time, and
    # the row size is D.
    mask = col_offsets < D

    input_ptr = tl.multiple_of(input_ptr, 16)
    bias_ptr = tl.multiple_of(bias_ptr, 16)
    y_ptr = tl.multiple_of(y_ptr, 16)

    offset = bt_idx * bt_stride
    buffer_ptrs = symm_mem_buffer_ptrs.to(tl.pointer_type(tl.uint64))

    # Copy the input, x, to the symmetric memory buffer.
    buffer_ptr = tl.load(buffer_ptrs + rank).to(tl.pointer_type(tl.bfloat16))
    buffer_ptr = tl.multiple_of(buffer_ptr, 16)
    for i in tl.static_range(rows_per_block):
        row = tl.load(input_ptr + offset + i * D + col_offsets, mask=mask)
        tl.store(buffer_ptr + offset + i * D + col_offsets, row, mask=mask)

    symm_mem_sync(
        symm_mem_signal_pad_ptrs,
        None,
        rank,
        world_size,
        hasPreviousMemAccess=True,
        hasSubsequentMemAccess=True,
    )

    # Two shot allreduce
    local_rank_offsets = (
        offset
        + size_per_rank * rank
        + tl.arange(0, triton.next_power_of_2(size_per_rank))
    )
    local_rank_mask = local_rank_offsets < (offset + size_per_rank * (rank + 1))

    # Bias addition, this is feasible because the bias is the same across ranks.
    acc = tl.load(bias_ptr + local_rank_offsets, mask=local_rank_mask).to(tl.float32)
    for remote_rank in range(world_size):
        buffer_ptr = tl.load(buffer_ptrs + remote_rank).to(tl.pointer_type(tl.bfloat16))
        buffer_ptr = tl.multiple_of(buffer_ptr, 16)
        val = tl.load(buffer_ptr + local_rank_offsets, mask=local_rank_mask).to(
            tl.float32
        )
        acc += val

    for remote_rank in range(world_size):
        buffer_ptr = tl.load(buffer_ptrs + remote_rank).to(tl.pointer_type(tl.bfloat16))
        buffer_ptr = tl.multiple_of(buffer_ptr, 16)
        tl.store(buffer_ptr + local_rank_offsets, acc, mask=local_rank_mask)

    symm_mem_sync(
        symm_mem_signal_pad_ptrs,
        None,
        rank,
        world_size,
        hasPreviousMemAccess=True,
        hasSubsequentMemAccess=True,
    )

    # The regular RMSNorm
    buffer_ptr = tl.load(buffer_ptrs + rank).to(tl.pointer_type(tl.bfloat16))
    buffer_ptr = tl.multiple_of(buffer_ptr, 16)
    for i in tl.static_range(rows_per_block):
        row = tl.load(buffer_ptr + offset + i * D + col_offsets, mask=mask).to(
            tl.float32
        )
        variance = tl.sum(row * row, axis=0) / D
        rstd = tl_rsqrt(variance + eps)

        w = tl.load(w_ptr + col_offsets, mask=mask).to(tl.float32)
        tl.store(y_ptr + offset + i * D + col_offsets, row * rstd * w, mask=mask)

    symm_mem_sync(
        symm_mem_signal_pad_ptrs,
        None,
        rank,
        world_size,
        hasPreviousMemAccess=True,
    )

def two_shot_all_reduce_bias_rms_norm(
    symm_mem_input: torch.Tensor,
    x: torch.Tensor,
    bias: torch.Tensor,
    rms_weight: torch.Tensor,
    eps: float = 1.0e-5,
    group: dist.ProcessGroup | None = None,
) -> None:
    """Performs two-shot all-reduce, bias addition and RMSNorm in a fused manner.

    This function executes the following operations:
    dist.all_reduce(x)
    x = x + bias
    y = F.rms_norm(x, x.shape[-1], w, eps)

    Args:
        symm_mem_input (torch.Tensor): The symmetric memory buffer for
            communication.
        x (torch.Tensor): The input tensor to be reduced.
        bias (torch.Tensor): The bias tensor to be added to the reduced input.
        w (torch.Tensor): The weights tensor for RMS normalization.
        y (torch.Tensor): The output tensor to store the result.
        eps (float, optional): The epsilon value for RMSNorm. Defaults to
            1.0e-5.
        group (dist.ProcessGroup, optional): The process group for allreduce.
            Defaults to None which uses the WORLD process group.

    Returns:
        None: The result is stored in the output tensor y.
    """
    w = rms_weight
    y = torch.empty_like(x)
    D = x.shape[-1]

    assert y.shape == x.shape
    assert y.dtype == x.dtype
    assert y.stride() == x.stride(), (str(x.stride()), str(y.stride()))
    assert w.is_contiguous()
    assert w.shape == (D,)
    assert x.is_contiguous()
    assert y.is_contiguous()

    total_rows = math.prod(x.shape[:-1])

    # We only support certain total_rows to just demonstrate the idea.
    BLOCK_SIZE = 1024
    if total_rows < 2:
        rows_per_block = 1
    elif total_rows <= 32:
        rows_per_block = 2
    elif total_rows <= 64:
        rows_per_block = 4
    else:
        rows_per_block = 8

    num_blocks = total_rows // rows_per_block

    num_warps = 32
    group = group or dist.group.WORLD
    symm_mem_hdl = symm_mem.rendezvous(symm_mem_input, group=group)
    world_size = symm_mem_hdl.world_size
    rank = symm_mem_hdl.rank

    size_per_rank = rows_per_block * D // world_size

    # assert total_rows % rows_per_block == 0
    assert size_per_rank * world_size == rows_per_block * D
    assert size_per_rank % 16 == 0

    kernel = two_shot_all_reduce_bias_rms_norm_kernel[(num_blocks,)](
        symm_mem_hdl.buffer_ptrs_dev,
        symm_mem_hdl.signal_pad_ptrs_dev,
        x,
        bias,
        w,
        y,
        eps,
        D=D,
        bt_stride=D,
        size_per_rank=size_per_rank,
        rows_per_block=rows_per_block,
        rank=rank,
        world_size=world_size,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=num_warps,
    )

    dump_kernel = os.environ.get("SYMM_DUMP_KERNEL", "0") == "1"
    if dump_kernel and torch.distributed.get_rank() == 0:
        log.log_triton_kernel(kernel)

    return y

def two_shot_all_reduce_bias_with_rms_norm(x, bias, rms_weight, symm_mem_input):
    y = torch.empty_like(x)
    two_shot_all_reduce_bias(symm_mem_input, x, bias, y)
    return fused_rms_norm(y, rms_weight)

def two_shot_all_reduce_bias(
    symm_mem_input: torch.Tensor,
    input_tensor: torch.Tensor,
    bias: torch.Tensor | None,
    output: torch.Tensor,
    max_num_blocks: int = 24,
    BLOCK_SIZE: int = 2048,
    group: dist.ProcessGroup | None = None,
):
    """
    Perform a two-shot all-reduce operation with bias addition using symmetric memory.

    output = all_reduce(input)
    output = output + bias if bias is not None else output

    NOTE: bias is the same across ranks for this use case as the workload is for inference.

    Args:
        symm_mem_input (torch.Tensor): The symmetric memory buffer.
        input_tensor (torch.Tensor): The input tensor to be reduced. Must be of dtype
            torch.bfloat16 and 128-bit aligned.
        bias (torch.Tensor | None): The bias tensor to be added to the reduced
            input. If None, no bias is added.
        output (torch.Tensor): The output tensor to store the result.
        max_num_blocks (int, optional): The maximum number of blocks to launch.
        BLOCK_SIZE (int, optional): The BLOCK_SIZE parameter for the kernel.
        group (dist.ProcessGroup | None, optional): The process group to use for
            the all-reduce operation. If None, the default process group will be
            used.

    Returns:
        torch.Tensor: The output tensor containing the reduced result with bias added.
    """

    group = group or dist.group.WORLD
    symm_mem_hdl = symm_mem.rendezvous(symm_mem_input, group=group)

    world_size = symm_mem_hdl.world_size
    num_blocks = min(
        triton.cdiv(input_tensor.numel(), BLOCK_SIZE * world_size), max_num_blocks
    )
    rank = symm_mem_hdl.rank

    assert input_tensor.dtype == torch.bfloat16, "Only bfloat16 is supported for now."
    assert input_tensor.numel() % 8 == 0, (
        "The number of elements must be 128-bit aligned."
    )
    assert BLOCK_SIZE % world_size == 0

    num_warps = 32

    kernel = two_shot_all_reduce_bias_kernel[(num_blocks,)](
        symm_mem_hdl.buffer_ptrs_dev,
        symm_mem_hdl.signal_pad_ptrs_dev,
        input_tensor,
        bias,
        output,
        numel=input_tensor.numel(),
        has_bias=bias is not None,
        stride_per_program=BLOCK_SIZE * world_size,
        rank=rank,
        world_size=world_size,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=num_warps,
    )

    dump_kernel = os.environ.get("SYMM_DUMP_KERNEL", "0") == "1"
    if dump_kernel and torch.distributed.get_rank() == 0:
        log.log_triton_kernel(kernel)

    return output

@triton.jit
def two_shot_all_reduce_bias_kernel(
    symm_mem_buffer_ptrs,
    symm_mem_signal_pad_ptrs,
    input_ptr,
    bias_ptr,
    output_ptr,
    numel,
    has_bias: tl.constexpr,
    stride_per_program: tl.constexpr,
    rank: tl.constexpr,
    world_size: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    input_ptr = tl.multiple_of(input_ptr, 16)
    output_ptr = tl.multiple_of(output_ptr, 16)
    buffer_ptrs = symm_mem_buffer_ptrs.to(tl.pointer_type(tl.uint64))
    # Note: Triton complains this is not a constexpr, but it is :(
    # stride_per_program = BLOCK_SIZE * world_size

    # Copy the input to the symmetric memory buffer.
    # Each PID needs to perform copy for every BLOCK_SIZE * world_size elements.
    buffer_ptr = tl.load(buffer_ptrs + rank).to(tl.pointer_type(tl.bfloat16))
    buffer_ptr = tl.multiple_of(buffer_ptr, 16)
    block_start = pid * stride_per_program
    while block_start < numel:
        offsets = block_start + tl.arange(0, tl.constexpr(stride_per_program))
        mask = offsets < numel
        val = tl.load(input_ptr + offsets, mask=mask)
        tl.store(buffer_ptr + offsets, val, mask=mask)
        block_start += tl.num_programs(axis=0) * stride_per_program

    symm_mem_sync(
        symm_mem_signal_pad_ptrs,
        None,
        rank,
        world_size,
        hasPreviousMemAccess=True,
        hasSubsequentMemAccess=True,
    )

    # Two-shot allreduce
    block_start = pid * stride_per_program
    while block_start < numel:
        offsets = block_start + rank * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < numel
        acc = tl.zeros([BLOCK_SIZE], dtype=tl.float32)

        for i in range(world_size):
            buffer_ptr = tl.load(buffer_ptrs + i).to(tl.pointer_type(tl.bfloat16))
            buffer_ptr = tl.multiple_of(buffer_ptr, 16)
            val = tl.load(buffer_ptr + offsets, mask=mask).to(tl.float32)
            acc += val

        # NOTE: Doing this between two shots is feasible because the bias is the
        # same across all ranks.
        if has_bias:
            bias_ptr = tl.multiple_of(bias_ptr, 16)
            acc += tl.load(bias_ptr + offsets, mask=mask).to(tl.float32)

        # XXX not synchronization is needed btw these 2 shots?
        for i in range(world_size):
            buffer_ptr = tl.load(buffer_ptrs + i).to(tl.pointer_type(tl.bfloat16))
            buffer_ptr = tl.multiple_of(buffer_ptr, 16)
            tl.store(buffer_ptr + offsets, acc, mask=mask)

        block_start += tl.num_programs(axis=0) * stride_per_program

    symm_mem_sync(
        symm_mem_signal_pad_ptrs,
        None,
        rank,
        world_size,
        hasPreviousMemAccess=True,
        hasSubsequentMemAccess=True,
    )

    # Copy the result from the symmetric memory buffer to the output.
    buffer_ptr = tl.load(buffer_ptrs + rank).to(tl.pointer_type(tl.bfloat16))
    buffer_ptr = tl.multiple_of(buffer_ptr, 16)
    block_start = pid * stride_per_program
    while block_start < numel:
        offsets = block_start + tl.arange(0, stride_per_program)
        mask = offsets < numel
        val = tl.load(buffer_ptr + offsets, mask=mask).to(tl.float32)
        tl.store(output_ptr + offsets, val, mask=mask)
        block_start += tl.num_programs(axis=0) * stride_per_program

    # Ensure that subsequent kernels do not corrupt the data before this kernel
    # completes loading from the symmetric memory.
    symm_mem_sync(
        symm_mem_signal_pad_ptrs,
        None,
        rank,
        world_size,
        hasPreviousMemAccess=True,
    )

def plot_experiment_comparison(
    sizes: list[str], experiments: list[str], data: list[list[float]], filename: str
):
    """
    Plots and saves a line chart comparing mechanisms' running times
    across experiment settings.

    Args:
        sizes: The input sizes.
        experiments: The names of the experiments.
        data: The data to plot.
        filename: The filename to save the plot to.
    """
    import matplotlib.pyplot as plt

    # Prepare X-axis labels
    x_pos = range(len(sizes))
    plt.figure(figsize=(12, 6))
    for i, experiment in enumerate(experiments):
        plt.plot(x_pos, [row[i] for row in data], marker="o", label=experiment)

    plt.xticks(x_pos, sizes, rotation=30, ha="right")
    plt.ylabel("Running Time (us)")
    plt.xlabel("Sizes")
    plt.title("Experiments")
    plt.legend()
    plt.grid(True, axis="y")
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()
    print(f"Plot saved as {filename}")

if __name__ == "__main__":
    main()
