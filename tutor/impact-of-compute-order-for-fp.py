"""
seed         -1, sum -856.79492188
seed         23, sum -856.79003906
seed         23, sum -856.79003906
seed       1337, sum -856.78906250
seed      12345, sum -856.79296875
seed     123456, sum -856.79199219
"""

import torch

N = 1024 * 1024 * 1024
torch.manual_seed(37)
x = torch.randn(N, device="cuda")

for s in [-1, 23, 23, 1337, 12345, 123456]:
    if s >= 0:
        torch.manual_seed(s)
        perm = torch.randperm(N, device="cuda")
    else:
        perm = torch.arange(N, device="cuda")

    y = x[perm]
    print(f"seed {s:10}, sum {y.sum().item():.8f}")
