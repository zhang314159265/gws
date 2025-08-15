import torch
from triton.testing import do_bench

libkernel = torch.library.Library("kernel", "FRAGMENT")

def register_custom_op(name):
    def f_meta(x):
        return torch.empty_like(x)

    def decorator(f):
        libkernel.define(f"{name}(Tensor self) -> Tensor", tags=())
        libkernel.impl(name, f, "CUDA")
        libkernel.impl(name, f_meta, "Meta")
        
        return f
    return decorator

def bench(f, label, total_bytes):
    for _ in range(5):  # warmup
        f()
    ms = do_bench(f)
    gbps = (total_bytes * 1e-9) / (ms * 1e-3)
    print(f"{label}: {ms:.3f}ms {gbps:.3f}gbps")

def checkclose(ref, act, tol=1e-5):
    assert torch.allclose(ref, act, atol=tol, rtol=tol), f"ref:\n{ref}\nact:\n{act}\n"
