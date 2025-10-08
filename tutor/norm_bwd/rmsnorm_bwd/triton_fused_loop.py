import torch
import triton
import triton.language as tl

from .utils import cdiv

@triton.jit
def kernel_fused(x, w, rsqrt, dy, dx, interm, M, N, MBLK: tl.constexpr, NBLK: tl.constexpr):
    midx = tl.program_id(0) * MBLK + tl.arange(0, MBLK)
    midx = midx[:, None]
    mmask = midx < M

    accum = tl.full([MBLK, NBLK], 0.0, tl.float32)
    rsqrtval = tl.load(rsqrt + midx, mask=mmask, other=0.0).to(tl.float32)
    for noff in range(0, N, NBLK):
        nidx = noff + tl.arange(0, NBLK)
        nidx = nidx[None, :]
        nmask = nidx < N

        # load
        dyval = tl.load(dy + midx * N + nidx, mask=mmask & nmask, other=0.0).to(tl.float32)
        xval = tl.load(x + midx * N + nidx, mask=mmask & nmask, other=0.0).to(tl.float32)
        wval = tl.load(w + nidx, mask=nmask, other=0.0).to(tl.float32)

        accum += dyval * xval * wval

        dwval = tl.sum(xval * rsqrtval * dyval, axis=0)[None, :]
        tl.store(interm + tl.program_id(0) * N + nidx, dwval, mask=nmask)

    sumval = tl.sum(accum, axis=1)[:, None]

    for noff in range(0, N, NBLK):
        nidx = noff + tl.arange(0, NBLK)
        nidx = nidx[None, :]
        nmask = nidx < N

        # load
        xval = tl.load(x + midx * N + nidx, mask=mmask & nmask, other=0.0).to(tl.float32)
        wval = tl.load(w + nidx, mask=nmask, other=0.0).to(tl.float32)
        dyval = tl.load(dy + midx * N + nidx, mask=mmask & nmask, other=0.0).to(tl.float32)

        # y = x * rsqrt * w
        dx_part1 = dyval * rsqrtval * wval
        dx_part2 = sumval * xval / N * rsqrtval * rsqrtval * rsqrtval
        dxval = dx_part1 - dx_part2
        tl.store(dx + midx * N + nidx, dxval, mask=mmask & nmask)


def triton_fused_loop_bwd(x, w, rsqrt, dy, _y_ignore):
    dx = torch.empty_like(x)
    MBLK = 256
    NBLK = 64
    nsplit = cdiv(x.size(0), MBLK)

    interm = torch.empty([nsplit, x.shape[1]], device=x.device, dtype=torch.float)

    kernel_fused[(nsplit,)](x, w, rsqrt, dy, dx, interm, x.shape[0], x.shape[1], MBLK, NBLK)
    dw = interm.sum(dim=0)

    return dx, dw
