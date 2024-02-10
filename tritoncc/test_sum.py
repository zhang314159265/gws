"""
ttir generated before and after make_ttir: https://gist.github.com/shunting314/b3b7c0ce246cc83ef09e5249853c3df4

After running make_ttgir, got module: https://gist.github.com/shunting314/485d4a5614203b2d3170a446daaba37c

After running the pass manager in make_llir, we get this module: https://gist.github.com/shunting314/454a632f06e2442dd67c8e93859538f7
"""
import torch
import triton
from triton import language as tl

# Use a persistent reduction
@triton.jit
def fn(iptr, optr, xnumel, rnumel, XBLOCK: tl.constexpr, RBLOCK: tl.constexpr):
    xidx = tl.program_id(0) * XBLOCK + tl.arange(0, XBLOCK)
    ridx = tl.arange(0, RBLOCK)
    data = tl.load(iptr + xidx[:, None] * rnumel + ridx[None, :], mask=(xidx[:, None] < xnumel) & (ridx[None, :] < rnumel))
    data = tl.sum(data, axis=-1)
    tl.store(optr + xidx, data, mask=(xidx < xnumel))

torch.set_default_device("cuda")
M = 1024
N = 1024
x = torch.rand(M, N)
act = torch.empty(M)
ref = torch.sum(x, dim=-1)

XBLOCK = 2
RBLOCK = triton.next_power_of_2(N)
fn[((M + XBLOCK - 1) // XBLOCK, 1, 1)](x, act, M, N, XBLOCK=XBLOCK, RBLOCK=RBLOCK)
assert torch.allclose(ref, act)
print("bye red")
