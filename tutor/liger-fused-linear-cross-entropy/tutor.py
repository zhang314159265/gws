import sys
import os
from torch.nn import functional as F

sys.path.append(os.path.dirname(__file__))
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import torch

from torch import nn
from utils import bench

sys.path.append("/home/shunting/ws/Liger-Kernel/src")
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

opt_torchf = torch.compile(torchf)

expected = torchf(model, x, T).float()
assert torch.allclose(expected, ligerf(model, x, T).float(), atol=1e-2, rtol=1e-2)
assert torch.allclose(expected, opt_torchf(model, x, T).float(), atol=1e-2, rtol=1e-2)

bench(lambda: ligerf(model, x, T), "liger")
bench(lambda: torchf(model, x, T), "torch")
bench(lambda: opt_torchf(model, x, T), "opt_torch")
print("bye")
