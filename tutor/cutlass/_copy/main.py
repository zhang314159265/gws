import torch
import functools
from triton.testing import do_bench
from .cutlass_basic import cutlass_basic
from .cutlass_vectorized import cutlass_vectorized

def bench(variant, fn, *, tot_nbytes, ref):
    act = fn()
    torch.testing.assert_close(ref, act)
    ms = do_bench(fn)
    tbgs = (tot_nbytes * 1e-12) / (ms * 1e-3)
    print(f"{variant:20}: {ms:.3f} ms, {tbgs:.3f} tbgs")

def ref_fn(x):
    y = torch.empty_like(x)
    y.copy_(x)
    return y

x = torch.randn(256, 512, device="cuda")
ref = ref_fn(x)

tot_nbytes = x.nbytes * 2
bench = functools.partial(bench, tot_nbytes=tot_nbytes, ref=ref)
bench("torch", lambda: ref_fn(x))
bench("cutlass_vectorized", lambda: cutlass_vectorized(x))
bench("cutlass_basic", lambda: cutlass_basic(x))
