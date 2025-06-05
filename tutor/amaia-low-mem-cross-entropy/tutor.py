"""
Ideally there should be only 2 compiled graph. But fwd/bwd are both being
re-compiled twice. There are in total 6 compiled graphs.
"""

import sys
import os
sys.path.append(os.path.dirname(__file__))
import torch
from cross_entropy import fused_matmul_cross_entropy
import torch.nn.functional as F

BT = 32768
V = 128256
D = 768
X = torch.randn(BT, D, device="cuda", dtype=torch.bfloat16, requires_grad=True)
W = torch.randn(V, D, device="cuda", dtype=torch.bfloat16, requires_grad=True)
T = torch.randint(0, V, (BT,), device="cuda")

loss = fused_matmul_cross_entropy(X, W, T).mean()
loss.backward()
print(W.grad)
expected = F.cross_entropy(X @ W.t(), T)
print(loss)
print(expected)

print("bye")
