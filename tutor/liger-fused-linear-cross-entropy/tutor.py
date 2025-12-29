import sys
import os
from torch.nn import functional as F

import torch

from torch import nn
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
        path = f"/tmp/{name}.json"
        prof.export_chrome_trace(path)
        print(f"Profile trace writen to {path}")
   
    torch.cuda.reset_peak_memory_stats()
    ms = do_bench(f)

    print(f"{name}: {ms:.3f}ms")
    print(f"Peak mem: {torch.cuda.max_memory_allocated() / 1e9:.3f} GB")
    print()


from liger_kernel.ops.fused_linear_cross_entropy import LigerFusedLinearCrossEntropyFunction

torch.set_default_device("cuda")

BT, C, V = 32768, 768, 128256
model = nn.Linear(C, V, bias=False).bfloat16()
x = torch.randn(BT, C, requires_grad=True, dtype=torch.bfloat16)
T = torch.randint(0, V, (BT,))

def ligerf(m, x, label):
    x.grad = None
    m.weight.grad = None

    out = LigerFusedLinearCrossEntropyFunction.apply(x, m.weight, label)[0]
    out.backward()
    return out

def torchf(m, x, label):
    x.grad = None
    m.weight.grad = None

    loss = F.cross_entropy(m(x), label)
    loss.backward()
    return loss

opt_torchf = torch.compile(torchf, options={"auto_chunker.enable": False})
compiled_ligerf = torch.compile(ligerf, options={"auto_chunker.enable": False})

expected = torchf(model, x, T).float()
assert torch.allclose(expected, ligerf(model, x, T).float(), atol=1e-2, rtol=1e-2)
assert torch.allclose(expected, compiled_ligerf(model, x, T).float(), atol=1e-2, rtol=1e-2)
assert torch.allclose(expected, opt_torchf(model, x, T).float(), atol=1e-2, rtol=1e-2)

bench(lambda: ligerf(model, x, T), "liger", profile=True)
bench(lambda: compiled_ligerf(model, x, T), "compiled_liger", profile=True)
bench(lambda: torchf(model, x, T), "torch-eager")
bench(lambda: opt_torchf(model, x, T), "compile_no_chunking")

for log_nchunk in range(2, 7):
    torch._dynamo.reset()
    nchunk = 2 ** log_nchunk
    do_profile = nchunk == 64

    # why num_chunk can not be ov
    autochunker_torchf = torch.compile(torchf, options={"auto_chunker.enable": True, "auto_chunker.num_chunk": nchunk})
    assert torch.allclose(expected, autochunker_torchf(model, x, T).float(), atol=1e-2, rtol=1e-2)
    bench(lambda: autochunker_torchf(model, x, T), f"compile_{nchunk}_chunks", profile=do_profile)

print("bye")
