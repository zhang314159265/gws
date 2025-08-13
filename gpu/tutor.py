"""
A tutorial to use curun
"""

import curun
import torch

mod = curun.open("add.cubin")
func = curun.sym(mod, "add")
a = torch.ones(1024, device="cuda", dtype=torch.bfloat16)
b = torch.ones(1024, device="cuda", dtype=torch.bfloat16) * 2
c = torch.empty(1024, device="cuda", dtype=torch.bfloat16)
ref = torch.ones(1024, device="cuda", dtype=torch.bfloat16) * 3

eq_before = torch.allclose(c, ref)

curun.run(
    func, 32, 1, 1, 32 * 4, 1, 1, 0, 0, [a.data_ptr(), b.data_ptr(), c.data_ptr(), a.numel()]
)

eq_after = torch.allclose(c, ref)

if not eq_before and eq_after:
    print("PASS")
else:
    print("FAIL")
