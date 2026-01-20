import torch
from triton.testing import do_bench
from .cuda_add import cuda_add
from .cutlass_add_basic import cutlass_add_basic
import functools

def bench(variant, fn, *, total_nbytes, ref):
    act = fn()
    torch.testing.assert_close(ref, act)
    ms = do_bench(fn)
    tbgs = (total_nbytes * 1e-12) / (ms * 1e-3)
    print(f"{variant:20}: {ms:.3f} ms, {tbgs: .3f} tbgs")

N = 1024 * 1024 * 1024

x = torch.randn(N, device="cuda")
y = torch.randn(N, device="cuda")

ref = x + y

total_nbytes = x.nbytes * 3
bench = functools.partial(bench, total_nbytes=total_nbytes, ref=ref)
bench("torch", lambda: x + y)
bench("cuda_add", lambda: cuda_add(x, y))
bench("cutlass_add_basic", lambda: cutlass_add_basic(x, y))
