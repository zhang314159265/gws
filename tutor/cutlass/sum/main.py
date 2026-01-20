import torch
from triton.testing import do_bench
import functools
from .cutlass_sum import cutlass_sum

def bench(variant, fn, *, ref, total_bytes):
    act = fn()
    torch.testing.assert_close(ref, act)
    ms = do_bench(fn)
    tbgs = (total_bytes * 1e-12) / (ms * 1e-3)
    print(f"{variant:20}: {ms:.3f} ms, {tbgs:.3f} tbgs")

x = torch.randn(1024, 512, device="cuda")
ref = x.sum(dim=-1)
total_bytes = x.nbytes + ref.nbytes
bench = functools.partial(bench, ref=ref, total_bytes=total_bytes)

bench("Eager", lambda: x.sum(dim=-1))
bench("cutlass_sum", lambda: cutlass_sum(x))
