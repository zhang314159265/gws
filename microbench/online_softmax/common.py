import torch
from triton.testing import do_bench
import os

def eager(inp):
    inp32 = inp.to(torch.float32)
    ref_max = inp32.amax(dim=-1, keepdim=True)
    ref_sum = (inp32 - ref_max).exp().sum(dim=-1, keepdim=True).log()
    return ref_max, ref_sum

def verify_numeric(inp, act_max, act_sum, ref_max=None, ref_sum=None):
    if ref_max is None or ref_sum is None:
        ref_max, ref_sum = eager(inp)

    tol = 1e-3
    return torch.allclose(ref_max, act_max, atol=tol, rtol=tol) and torch.allclose(ref_sum, act_sum, atol=tol, rtol=tol)

FILTER = os.environ.get("FILTER", "")

def benchmark(label, call, args, ref_max=None, ref_sum=None):
    if FILTER and label not in FILTER:
        return
    args[1].zero_()
    args[2].zero_()
    ms = do_bench(lambda: call(args))
    status = "PASS" if verify_numeric(*args, ref_max, ref_sum) else "FAIL"
    print(f"{label:20s} {ms:.3f}ms {status}")
