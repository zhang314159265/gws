"""
A tutorial to use curun
"""

import curun
import torch

a = torch.ones(1024, device="cuda", dtype=torch.bfloat16)
b = torch.ones(1024, device="cuda", dtype=torch.bfloat16) * 2
c = torch.empty(1024, device="cuda", dtype=torch.bfloat16)
ref = torch.ones(1024, device="cuda", dtype=torch.bfloat16) * 3

eq_before = torch.allclose(c, ref)

curun.open("add.cubin").sym("add")[32, 32 * 4](a, b, c, a.numel())

eq_after = torch.allclose(c, ref)

if not eq_before and eq_after:
    print("PASS")
else:
    print("FAIL")
