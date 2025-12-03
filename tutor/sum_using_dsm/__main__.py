import torch
from triton.testing import do_bench
from .cuda import cuda_sum, cuda_sum_dsm

x = torch.randn(256, 128256, dtype=torch.float, device="cuda")
ref = x.sum(dim=-1, keepdim=True) + x

total_bytes = x.nbytes + ref.nbytes

def check_and_bench(fn, name=None):
    if not name:
        name = fn.__name__
    act = fn() 
    torch.testing.assert_close(ref, act, atol=1e-3, rtol=1e-3)

    ms = do_bench(fn) 
    tbps = total_bytes * 1e-12 / (ms * 1e-3)
    print(f"{name}: {ms:.3f} ms, {tbps:.3f} tbps")

@torch.compile
def compiled_sum(x):
    return x.sum(dim=-1, keepdim=True) + x

for _ in range(5): # warmup
    compiled_sum(x)

check_and_bench(lambda: x.sum(dim=-1, keepdim=True) + x, name="eager")
check_and_bench(lambda: compiled_sum(x), name="compiled")
check_and_bench(lambda: cuda_sum(x), name="cuda")
check_and_bench(lambda: cuda_sum_dsm(x), name="cuda with dsm")
