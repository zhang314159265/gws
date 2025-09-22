import cutlass
from cutlass import cute
from cutlass.cute.runtime import from_dlpack
import torch
import numpy as np
from triton.testing import do_bench

N = 1024 * 1024 * 1024
x = torch.randn(N, device="cuda")
ref = x.relu()
act = torch.empty(N, device="cuda")

def benchmark(fn):
    nbytes = x.numel() * x.itemsize * 2
    ms = do_bench(fn)
    gbps = nbytes * 1e-9 / (ms * 1e-3)
    print(f"{ms=:.3f} {gbps=:.3f}")

@cute.kernel
def cutedsl_relu_kernel(x, y):
    tidx, _, _ = cute.arch.thread_idx()
    bidx, _, _ = cute.arch.block_idx()
    bdim, _, _ = cute.arch.block_dim()

    elemidx = bidx * bdim + tidx

    xval = x[elemidx]
    yval = xval if xval >= 0 else 0.0
    y[elemidx] = yval
    # if tidx == 0:
    #     cute.printf("{} {}", xval, yval)

@cute.jit
def cutedsl_relu(x, y):
    kernel = cutedsl_relu_kernel(x, y)

    block_size = 128
    numel = cute.size(x)
    assert numel % block_size == 0, f"{numel=}"

    kernel(grid=(numel // block_size, 1, 1), block=(block_size, 1, 1))

@cute.kernel
def cutedsl_vectorize_relu_kernel(x, y):
    tidx, _, _ = cute.arch.thread_idx()
    bidx, _, _ = cute.arch.block_idx()
    bdim, _, _ = cute.arch.block_dim()

    gtidx = bidx * bdim + tidx

    xval = x[None, gtidx].load() # use None rather than : to load the entire dimension
    
    # why > 0 works while >= 0 not!
    cond = xval > 0
    yval = cute.where(cond, xval, cute.full_like(xval, 0))
    y[None, gtidx] = yval


@cute.jit
def cutedsl_vectorize_relu(x, y):
    tile_size = 4
    x = cute.flat_divide(x, (tile_size,))
    y = cute.flat_divide(y, (tile_size,))
    block_size = 128
    nthreads = cute.size(x, mode=[1])
    assert nthreads % block_size == 0
    cutedsl_vectorize_relu_kernel(x, y)(
        grid=(nthreads // block_size, 1, 1),
        block=(block_size, 1, 1),
    )

x_ = from_dlpack(x)
act_ = from_dlpack(act)
cutedsl_relu_ = cute.compile(cutedsl_relu, x_, act_)
cutedsl_relu_(x_, act_)
torch.testing.assert_close(ref, act)

print("without vectorization:")
benchmark(lambda: cutedsl_relu_(x_, act_))

act.fill_(0)
cutedsl_vectorize_relu_ = cute.compile(cutedsl_vectorize_relu, x_, act_)
cutedsl_vectorize_relu_(x_, act_)
torch.testing.assert_close(ref, act)

print("with vectorization")
benchmark(lambda: cutedsl_vectorize_relu_(x_, act_))

print("PASS")
