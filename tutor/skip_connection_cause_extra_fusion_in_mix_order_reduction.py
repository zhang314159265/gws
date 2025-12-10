"""
The generated MixOrderReduction kernel fuses 2 outer reductions for the two
RMSNormBwd due to skip connections.
"""

import torch
import torch.nn.functional as F
from torch import nn

torch.set_default_device("cuda")

M, N = 8192, 4096
nlayer = 2

norms = [nn.RMSNorm(N) for _ in range(nlayer)]
linears = [nn.Linear(N, N, bias=False) for _ in range(nlayer)]

@torch.compile
def f(x):
    for i in range(nlayer):
        x = x + linears[i](norms[i](x))
        # x = linears[i](norms[i](x))
    return x

x = torch.randn(M, N, requires_grad=True)
f(x).sum().backward()
