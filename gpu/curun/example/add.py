"""
A tutorial to use curun
"""

import curun
import torch
from pathlib import Path

a = torch.ones(1024, device="cuda", dtype=torch.bfloat16)
b = torch.ones(1024, device="cuda", dtype=torch.bfloat16) * 2
c = torch.empty(1024, device="cuda", dtype=torch.bfloat16)
ref = torch.ones(1024, device="cuda", dtype=torch.bfloat16) * 3

eq_before = torch.allclose(c, ref)

add_cu_path = Path(__file__).parent / "add.cu"
curun.open(str(add_cu_path)).sym("add")[32, 32 * 4](a, b, c, a.numel())

eq_after = torch.allclose(c, ref)

if not eq_before and eq_after:
    print("PASS")
else:
    print("FAIL")
