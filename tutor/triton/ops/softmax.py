"""
Why the triton version is so much slower:
  Torch latency 7.951ms, triton latency 15.509ms
"""

import torch
import triton
import triton.language as tl

from common import bench

M = 32 * 1024 + 1 # +1 on purpose
N = 52001

@triton.jit
def kernel(x, y, xnumel, rnumel, XBLK: tl.constexpr):
    RBLK: tl.constexpr = 1024

    xidx = tl.program_id(0) * XBLK + tl.arange(0, XBLK)[:, None]
    xmask = xidx < xnumel

    rbase = tl.arange(0, RBLK)[None, :]
    max_accum = tl.full([XBLK, RBLK], float("-inf"), tl.float32)
    for roff in range(0, rnumel, RBLK):
        ridx = rbase + roff
        rmask = ridx < rnumel

        xval = tl.load(x + xidx * rnumel + ridx, xmask and rmask, other=float("-inf"))
        max_accum = tl.where(max_accum > xval, max_accum, xval)
    row_max = tl.max(max_accum, 1)[:, None]

    sum_accum = tl.full([XBLK, RBLK], 0, tl.float32)
    for roff in range(0, rnumel, RBLK):
        ridx = rbase + roff
        rmask = ridx < rnumel

        xval = tl.load(x + xidx * rnumel + ridx, xmask and rmask, other=0.0)
        sum_accum = sum_accum + tl.exp(xval - row_max)

    row_sum = tl.sum(sum_accum, 1)[:, None]

    for roff in range(0, rnumel, RBLK):
        ridx = rbase + roff
        rmask = ridx < rnumel

        # 'other' arg does not matter
        xval = tl.load(x + xidx * rnumel + ridx, xmask and rmask)
        yval = tl.exp(xval - row_max) / row_sum
        tl.store(y + xidx * rnumel + ridx, yval, xmask and rmask)


def launch(x):
    y = torch.empty_like(x)
    XBLK = 2
    kernel[(triton.cdiv(x.size(0), XBLK),)](x, y, x.size(0), x.size(1), XBLK)
    return y

x = torch.randn(M, N, device="cuda")
bench(lambda x: x.softmax(dim=-1), launch, (x,))
