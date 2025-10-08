import triton
import torch
import triton.language as tl
from .utils import cdiv, next_power_of_two

@triton.jit
def kernel_dx(x, w, rsqrt, dy, dx, N, R, XBLOCK: tl.constexpr, RBLOCK: tl.constexpr):
    xidx = (tl.program_id(0) * XBLOCK + tl.arange(0, XBLOCK))[:, None]
    ridx = tl.arange(0, RBLOCK)[None, :]
    xmask = xidx < N
    rmask = ridx < R

    # load
    xval = tl.load(x + xidx * R + ridx, mask=xmask & rmask, other=0.0).to(tl.float32)
    wval = tl.load(w + ridx, mask=rmask, other=0.0).to(tl.float32)
    rsqrtval = tl.load(rsqrt + xidx, mask=xmask, other=0.0).to(tl.float32)
    dyval = tl.load(dy + xidx * R + ridx, mask=xmask & rmask, other=0.0).to(tl.float32)

    # y = x * rsqrt * w
    dx_part1 = dyval * rsqrtval * wval
    dx_part2 = tl.sum(dyval * xval * wval, axis=1)[:, None] * xval / R * rsqrtval * rsqrtval * rsqrtval
    dxval = dx_part1 - dx_part2
    tl.store(dx + xidx * R + ridx, dxval, mask=xmask & rmask)

@triton.jit
def kernel_dw(x, rsqrt, dy, dw, N, M, RBLOCK: tl.constexpr):
    nidx = tl.program_id(0)

    accum = tl.full([RBLOCK], 0.0, tl.float32)
    for moff in range(0, M, RBLOCK):
        midx = moff + tl.arange(0, RBLOCK)
        mmask = midx < M

        xval = tl.load(x + nidx + midx * N, mask=mmask, other=0.0)
        rsqrtval = tl.load(rsqrt + midx, mask=mmask, other=0.0)
        dyval = tl.load(dy + nidx + midx * N, mask=mmask, other=0.0)

        tmp = xval * rsqrtval * dyval
        accum = accum + tmp

    dwval = tl.sum(accum)
    tl.store(dw + nidx, dwval)

def triton_bwd(x, w, rsqrt, dy, _y_ignore):
    dx = torch.empty_like(x)
    dw = torch.empty_like(w)

    # compute dx
    XBLOCK_dx = 32
    kernel_dx[(cdiv(x.size(0), XBLOCK_dx),)](x, w, rsqrt, dy, dx, x.size(0), x.size(1), XBLOCK=XBLOCK_dx, RBLOCK=next_power_of_two(x.size(1)))

    # compute dw
    kernel_dw[(x.size(1),)](x, rsqrt, dy, dw, x.size(1), x.size(0), RBLOCK=512)

    return dx, dw
