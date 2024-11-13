import torch
import triton
import triton.language as tl
import functools

from common import bench

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


@triton.jit
def blk_ptr_kernel(x, y, xnumel, rnumel, XBLK: tl.constexpr):
    RBLK: tl.constexpr = 512

    xoff = tl.program_id(0) * XBLK
    xidx = xoff + tl.arange(0, XBLK)

    # both order [0, 1] and [1, 0] works with the same perf..
    x_ptr = tl.make_block_ptr(x, [xnumel, rnumel], [rnumel, 1], [xoff, 0], [XBLK, RBLK], [1, 0])

    accum = tl.full([XBLK, RBLK], 0, dtype=tl.float32)
    for roff in range(0, rnumel, RBLK):
        xval = tl.load(x_ptr, boundary_check=[0, 1], padding_option="zero")
        accum = accum + xval

        x_ptr = tl.advance(x_ptr, [0, RBLK])

    yval = tl.sum(accum, 1)

    # y_ptr = tl.make_block_ptr(y, [xnumel], [1], [0], [XBLK], [0])
    # tl.store(y_ptr, yval)
    # XXX Use plain ptr for store for now
    tl.store(y + xidx, yval, xidx < xnumel)


def launch(kernel, x):
    # torch.empty may return the previously released memory without changing.
    # Use zeros instead.
    y = torch.zeros(x.size(0), device="cuda") 

    XBLK = 2
    kernel[(triton.cdiv(x.size(0), XBLK),)](x, y, x.size(0), x.size(1), XBLK)
    return y


M = 1024 * 512 + 1
N = 2048 + 1 # +1 on purpose
x = torch.randn(M, N, device="cuda")

print("Benchmark the block-ptr kernel")
bench(lambda x: x.sum(dim=-1), functools.partial(launch, blk_ptr_kernel), (x,))

print("Benchmark the persistent reduction")
bench(lambda x: x.sum(dim=-1), functools.partial(launch, persistent_reduction_kernel), (x,))

print("Benchmark the non-persistent reduction")
bench(lambda x: x.sum(dim=-1), functools.partial(launch, non_persistent_reduction_kernel), (x,))
