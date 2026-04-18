import torch
from torch import nn
import helion
import helion.language as hl

@helion.kernel(config=helion.Config(block_sizes=[64]), torch_compile_fusion=True)
def kernel(a, b):
    c = torch.empty_like(a)
    for t in hl.tile(a.shape):
        c[t] = a[t] + b[t]
    return c

@torch.compile
def f(a, b):
    a = a + 1
    b = b - 2
    c = kernel(a, b)
    d = c * 3
    return d

a, b = [torch.randn(1024, device="cuda") for _ in range(2)]
f(a, b)
