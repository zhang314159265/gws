import torch
from triton.testing import do_bench
from .cuda import cuda_sum, cuda_sum_dsm
import inspect

torch.manual_seed(1337)

dtype = torch.bfloat16 # support float and bfloat16
# dtype = torch.float
x = torch.randn(256, 128256, dtype=dtype, device="cuda")
y = torch.empty_like(x)
ref = x.sum(dim=-1, keepdim=True) + x

total_bytes = x.nbytes + ref.nbytes

def check_and_bench(fn, name=None):
    if not name:
        name = fn.__name__

    y.fill_(0);
    act = fn() 
    tol = 1e-3 if dtype == torch.float else 1e-2
    torch.testing.assert_close(ref, act, atol=tol, rtol=tol)

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
check_and_bench(lambda: cuda_sum(x, y), name="cuda")
check_and_bench(lambda: cuda_sum_dsm(x, y), name="cuda with dsm")
