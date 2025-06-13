"""
Ideally there should be only 2 compiled graph. But fwd/bwd are both being
re-compiled twice. There are in total 6 compiled graphs.
"""

import sys
import os
import torch
from torch import nn
import triton
from triton.testing import do_bench
sys.path.append(os.path.dirname(__file__))
from cross_entropy import fused_matmul_cross_entropy
import torch.nn.functional as F
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from utils import bench

BT = 32768
V = 128256
D = 768
X = torch.randn(BT, D, device="cuda", dtype=torch.bfloat16, requires_grad=True)
W = torch.randn(V, D, device="cuda", dtype=torch.bfloat16, requires_grad=True)
T = torch.randint(0, V, (BT,), device="cuda")

def amaia_f(x, y):
    X.grad = None
    W.grad = None

    loss = fused_matmul_cross_entropy(X, W, T).mean()
    loss.backward()
    return loss

# expected = F.cross_entropy(X @ W.t(), T)

linear = nn.Linear(D, V, bias=False).bfloat16().cuda()
def torch_f(x, y):
    linear.weight.grad = None
    x.grad = None

    loss = F.cross_entropy(linear(x).view(BT, -1), y.view(-1))
    loss.backward()
    return loss

bench(lambda: torch_f(X, T), name="torch")
bench(lambda: amaia_f(X, T), name="amaia")
opt_f = torch.compile(torch_f)
bench(lambda: opt_f(X, T), name="torch.compile", profile=True, profile_mem=True)

print("bye")
