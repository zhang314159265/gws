import triton
import torch
import triton.language as tl
from ..utils import cdiv, next_power_of_two

@triton.jit
def kernel_dx(x, w, mean, rstd, dy, dx, N, R, XBLOCK: tl.constexpr, RBLOCK: tl.constexpr):
    xidx = (tl.program_id(0) * XBLOCK + tl.arange(0, XBLOCK))[:, None]
    ridx = tl.arange(0, RBLOCK)[None, :]
    xmask = xidx < N
    rmask = ridx < R

    # load
    xval = tl.load(x + xidx * R + ridx, mask=xmask & rmask, other=0.0).to(tl.float32)
    dyval = tl.load(dy + xidx * R + ridx, mask=xmask & rmask, other=0.0).to(tl.float32)
    wval = tl.load(w + ridx, mask=rmask, other=0.0).to(tl.float32)
    meanval = tl.load(mean + xidx, mask=xmask, other=0.0).to(tl.float32)
    rstdval = tl.load(rstd + xidx, mask=xmask, other=0.0).to(tl.float32)

    # compute
    norm = (xval - meanval) * rstdval
    gnorm = dyval * wval
    dxval = gnorm
    dxval_part2 = tl.sum(gnorm, axis=1)[:, None] / R
    dxval_part3 = tl.sum(gnorm * norm, axis=1)[:, None] / R * norm
    dxval -= dxval_part2
    dxval -= dxval_part3
    dxval = dxval * rstdval

    # store
    tl.store(dx + xidx * R + ridx, dxval, mask=xmask & rmask)

@triton.jit
def kernel_dw_db(x, mean, rstd, dy, dw, db, N, M, RBLOCK: tl.constexpr):
    nidx = tl.program_id(0)

    accum_w = tl.full([RBLOCK], 0.0, tl.float32)
    accum_b = tl.full([RBLOCK], 0.0, tl.float32)
    for moff in range(0, M, RBLOCK):
        midx = moff + tl.arange(0, RBLOCK)
        mmask = midx < M

        xval = tl.load(x + nidx + midx * N, mask=mmask, other=0.0)
        meanval = tl.load(mean + midx, mask=mmask, other=0.0)
        rstdval = tl.load(rstd + midx, mask=mmask, other=0.0)
        dyval = tl.load(dy + nidx + midx * N, mask=mmask, other=0.0)

        accum_b = accum_b + dyval

        tmp = (xval - meanval) * rstdval * dyval
        accum_w = accum_w + tmp

    dwval = tl.sum(accum_w)
    dbval = tl.sum(accum_b)
    tl.store(dw + nidx, dwval)
    tl.store(db + nidx, dbval)


def triton_bwd(x, w, b, mean, rstd, dy, _y_ignore):
    dx = torch.empty_like(x)
    dw = torch.empty_like(w)
    db = torch.empty_like(b)

    XBLOCK_dx = 2
    kernel_dx[(cdiv(x.size(0), XBLOCK_dx),)](
        x, w, mean, rstd, dy, dx,
        x.size(0), x.size(1), XBLOCK=XBLOCK_dx, RBLOCK=next_power_of_two(x.size(1)))

    kernel_dw_db[(x.size(1),)](x, mean, rstd, dy, dw, db, x.size(1), x.size(0), RBLOCK=512)
    return dx, dw, db
