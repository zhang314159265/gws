import torch
from triton.testing import do_bench

def assert_close(ref, act, atol=1e-2, rtol=1e-2):
    assert len(ref) == 2
    torch.testing.assert_close(ref[0], act[0], atol=atol, rtol=rtol)
    torch.testing.assert_close(ref[1], act[1], atol=atol, rtol=rtol)

def bench(name, fn, *, total_bytes):
    ms = do_bench(fn) 
    tbps = total_bytes * 1e-12 / (ms * 1e-3)
    print(f"{name}: {ms:.3f} ms, {tbps:.3f} tbps")
