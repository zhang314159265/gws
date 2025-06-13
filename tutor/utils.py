import torch
from triton.testing import do_bench

def bench(f, name, warmup=5, profile_mem=False, profile=False):
    for _ in range(warmup):
        f()

    if profile_mem:
        torch.cuda.memory._record_memory_history()
        f()
        torch.cuda.memory._dump_snapshot(f"{name}.pickle")

    if profile:
        with torch.profiler.profile() as prof:
            f()
        prof.export_chrome_trace(f"{name}.json")
   
    torch.cuda.reset_peak_memory_stats()
    ms = do_bench(f)

    print(f"{name}: {ms:.3f}ms")
    print("Peak mem: ", torch.cuda.max_memory_allocated() / 1e9)
    print()


