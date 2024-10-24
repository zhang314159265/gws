import torch
import triton
import triton.language as tl
import functools

from common import bench

M = 1024 * 512
N = 2048 + 1 # +1 on purpose

# Use a persistent reduction
# XXX HACK: I mark rnumel as constexpr since it's used in tl.arange
@triton.jit
def persistent_reduction_kernel(x, y, xnumel, rnumel: tl.constexpr, XBLK: tl.constexpr):
    # XXX I'm surprised that triton.next_power_of_2 works in
    # the kernel. How does it work?
    RBLK: tl.constexpr = triton.next_power_of_2(rnumel)

    xoff = tl.program_id(0) * XBLK + tl.arange(0, XBLK)[:, None]
    roff = tl.arange(0, RBLK)[None, :]
    xval = tl.load(x + xoff * rnumel + roff, xoff < xnumel and roff < rnumel)
    # why this not work
    # https://gist.github.com/shunting314/1f23f33032ea3d90cc1e2f75eb89d77f
    # yval = xval.sum(axis=1, keep_dims=True)

    yval = tl.sum(xval, axis=1, keep_dims=True)
    tl.store(y + xoff, yval, xoff < xnumel)

# Use a non-persistent kernel
@triton.jit
def non_persistent_reduction_kernel(x, y, xnumel, rnumel, XBLK: tl.constexpr):
    xidx = tl.program_id(0) * XBLK + tl.arange(0, XBLK)[:, None]
    xmask = xidx < xnumel

    RBLK: tl.constexpr = 512

    rbase = tl.arange(0, RBLK)[None, :]
    # XXX Need use tl.float32 rather than torch.float32
    accum = tl.full([XBLK, RBLK], 0, dtype=tl.float32)
    for roff in range(0, rnumel, RBLK):
        ridx = rbase + roff 
        rmask = ridx < rnumel

        xval = tl.load(x + xidx * rnumel + ridx, xmask and rmask)
        accum = accum + xval

    yval = tl.sum(accum, axis=1)[:, None]
    tl.store(y + xidx, yval, xmask)

def launch(x, use_persistent_reduction):
    # torch.empty may return the previously released memory without changing.
    # Use zeros instead.
    y = torch.zeros(M, device="cuda") 

    XBLK = 2
    kernel = persistent_reduction_kernel if use_persistent_reduction else  non_persistent_reduction_kernel
    kernel[(triton.cdiv(x.size(0), XBLK),)](x, y, x.size(0), x.size(1), XBLK)
    return y


x = torch.randn(M, N, device="cuda")

print("Benchmark the persistent reduction")
bench(lambda x: x.sum(dim=-1), functools.partial(launch, use_persistent_reduction=True), (x,))

print("Benchmark the non-persistent reduction")
bench(lambda x: x.sum(dim=-1), functools.partial(launch, use_persistent_reduction=False), (x,))
