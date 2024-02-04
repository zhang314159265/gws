"""
A simple program to exercise the dot+add combination optimization happen in
make_ttir stage.

ttir dump before and after make_ttir stage:
https://gist.github.com/shunting314/6c3d560e1ac58899381cbf8092734f3c 
"""

import torch
import triton
from triton import language as tl
   
@triton.jit
def fn(aptr, bptr, optr, BLOCK: tl.constexpr):
    row = tl.arange(0, BLOCK)[:, None]
    col = tl.arange(0, BLOCK)[None, :]
    off = row * BLOCK + col
    lhs = tl.load(aptr + off)
    rhs = tl.load(bptr + off)
    out = tl.dot(lhs, rhs) + 5
    tl.store(optr + off, out)

torch.set_default_device("cuda")
N = 32
a = torch.rand(N, N)
b = torch.rand(N, N)
act = torch.rand(N, N)

fn[(1, 1, 1)](a, b, act, BLOCK=N)
ref = torch.mm(a, b) + 5
   
tol = 1e-3
assert torch.allclose(ref, act, atol=tol, rtol=tol), f"ref:\n{ref}\nact:\n{act}"
print("bye dot")
