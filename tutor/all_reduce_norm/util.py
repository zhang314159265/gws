import torch
from typing import Callable, Any
from contextlib import nullcontext
import torch.distributed as dist

def do_benchmark_with_event(
    target_fn: Callable[..., Any],
    ref: torch.Tensor | None,
    warmup_iters: int = 200,
    benchmark_iters: int = 200,
    flush_l2: bool = True,
    tol: float = 0.05,
    profile_ranks: list[int] | None = None,
) -> float:
    act = target_fn()
    if ref is not None:
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


