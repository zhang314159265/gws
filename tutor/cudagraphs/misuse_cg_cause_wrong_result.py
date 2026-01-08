"""
CUDAGraph cause numerical issues here because the 'extra'
tensor used during capturing, is not the 'extra' tensor we
are suppose to use during replaying.

The fix is use a persistent address for the extra tensor.

This is how full CG fail for flex-attn/decoding in vllm intiially.
"""

import torch

g = torch.cuda.CUDAGraph()

x = torch.randn(1024, device="cuda")
y = torch.randn(1024, device="cuda")
extra = torch.randn(1024, device="cuda")
with torch.cuda.graph(g):
    out = x + y + extra

a = torch.randn(1024, device="cuda")
b = torch.randn(1024, device="cuda")
extra = torch.randn(1024, device="cuda")
x.copy_(a)
y.copy_(b)
g.replay()
torch.testing.assert_close(out, a + b + extra)
print("bye")
