import torch
import functools
from triton.testing import do_bench
from .cutlass_matmul import cutlass_matmul
from .cutlass_tiled_matmul import cutlass_tiled_matmul

def bench(variant, fn, *, nbytes, nflops, ref):
    act = fn()
    torch.testing.assert_close(ref, act)
    ms = do_bench(fn)
    tbps = (nbytes * 1e-12) / (ms * 1e-3)
    tflopsps = (nflops * 1e-12) / (ms * 1e-3)
    print(f"{variant:20}: {ms:.3f} ms, {tbps:.3f} tb/s, {tflopsps:.3f} tflops/s")

M, N, K = 5120, 5120, 4096
x = torch.randn(M, K, device="cuda")
y = torch.randn(K, N, device="cuda")

ref = x @ y
nbytes = x.nbytes + y.nbytes + ref.nbytes
nflops = M * N * K * 2
bench = functools.partial(bench, nbytes=nbytes, nflops=nflops, ref=ref)
bench("torch", lambda: x @ y)
bench("cutlass_tiled_matmul", lambda: cutlass_tiled_matmul(x, y))
bench("cutlass_matmul", lambda: cutlass_matmul(x, y))
