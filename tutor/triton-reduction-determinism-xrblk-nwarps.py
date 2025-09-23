"""
Output: https://gist.github.com/shunting314/268f6ce3173f6294f41dca7d1d378516
"""
import torch
import triton
import triton.language as tl
import tabulate

@triton.jit
def kernel(x, y, xnumel, rnumel, XBLK: tl.constexpr, RBLK: tl.constexpr):
    xidx = tl.program_id(0) * XBLK + tl.arange(0, XBLK)[:, None]
    xmask = xidx < xnumel

    rbase = tl.arange(0, RBLK)[None, :]
    accum = tl.full([XBLK, RBLK], 0, dtype=tl.float32)
    for roff in range(0, rnumel, RBLK):
        ridx = rbase + roff 
        rmask = ridx < rnumel

        xval = tl.load(x + xidx * rnumel + ridx, xmask and rmask)
        accum = accum + xval

    yval = tl.sum(accum, axis=1)[:, None]
    tl.store(y + xidx, yval, xmask)

def launch(xblk, rblk, num_warps):
    y = torch.zeros(x.size(0), device="cuda")
    kernel[(triton.cdiv(x.size(0), xblk),)](x, y, x.size(0), x.size(1), XBLK=xblk, RBLK=rblk, num_warps=num_warps)
    return y

M, N = 1024, 1024
x = torch.randn(M, N, device="cuda")
ref = x.sum(dim=-1)

act = launch(2, 256, 4)

print("BITWISE EQUAL WITH EAGER?", torch.equal(ref, act))

def make_table(title, ys):
    tab = []
    for y1 in ys:
        tab.append([])
        for y2 in ys:
            tab[-1].append(str(torch.equal(y1, y2))[0])
    print(title)
    print(tabulate.tabulate(tab))

ylist = [launch(xblk=1, rblk=rblk, num_warps=4) for rblk in (32, 64, 128, 256, 512, 1024)]
make_table("CHANGE RBLOCK", ylist)

# num_warps 16, 32 generate same result because rblk is too small. Double
# the rblk will result in different numerics.
ylist = [launch(xblk=1, rblk=512, num_warps=num_warps) for num_warps in (2, 4, 8, 16, 32)]
make_table("CHANGE NUM_WARPS", ylist)


ylist = [launch(xblk=xblk, rblk=512, num_warps=4) for xblk in (1, 2, 4, 8, 16, 32, 64, 128, 256, 512)]
make_table("CHANGE XBLOCK", ylist)
