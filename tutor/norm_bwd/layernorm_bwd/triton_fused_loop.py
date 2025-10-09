import torch
import triton
import triton.language as tl

from ..utils import cdiv

@triton.jit
def kernel_fused(x, w, mean, rstd, dy, dx, interm_w, interm_b, M, N, MBLK: tl.constexpr, NBLK: tl.constexpr):
    midx = tl.program_id(0) * MBLK + tl.arange(0, MBLK)
    midx = midx[:, None]
    mmask = midx < M

    accum_gnorm = tl.full([MBLK, NBLK], 0.0, tl.float32)
    accum_gnorm_norm = tl.full([MBLK, NBLK], 0.0, tl.float32)
    rstdval = tl.load(rstd + midx, mask=mmask, other=0.0).to(tl.float32)
    meanval = tl.load(mean + midx, mask=mmask, other=0.0).to(tl.float32)
    for noff in range(0, N, NBLK):
        nidx = noff + tl.arange(0, NBLK)
        nidx = nidx[None, :]
        nmask = nidx < N

        # load
        dyval = tl.load(dy + midx * N + nidx, mask=mmask & nmask, other=0.0).to(tl.float32)
        xval = tl.load(x + midx * N + nidx, mask=mmask & nmask, other=0.0).to(tl.float32)
        wval = tl.load(w + nidx, mask=nmask, other=0.0).to(tl.float32)

        gnorm = dyval * wval
        norm = (xval - meanval) * rstdval
        accum_gnorm += gnorm
        accum_gnorm_norm += gnorm * norm

        # handle interm
        dwval = tl.sum(dyval * norm, axis=0)[None, :]
        dbval = tl.sum(dyval, axis=0)[None, :]
        tl.store(interm_w + tl.program_id(0) * N + nidx, dwval, mask=nmask)
        tl.store(interm_b + tl.program_id(0) * N + nidx, dbval, mask=nmask)

    mean_gnorm = tl.sum(accum_gnorm, axis=1)[:, None] / N
    mean_gnorm_norm = tl.sum(accum_gnorm_norm, axis=1)[:, None] / N

    for noff in range(0, N, NBLK):
        nidx = noff + tl.arange(0, NBLK)
        nidx = nidx[None, :]
        nmask = nidx < N

        # load
        xval = tl.load(x + midx * N + nidx, mask=mmask & nmask, other=0.0).to(tl.float32)
        wval = tl.load(w + nidx, mask=nmask, other=0.0).to(tl.float32)
        dyval = tl.load(dy + midx * N + nidx, mask=mmask & nmask, other=0.0).to(tl.float32)

        norm = (xval - meanval) * rstdval
        gnorm = dyval * wval
        dxval = gnorm - mean_gnorm - mean_gnorm_norm * norm
        dxval *= rstdval

        tl.store(dx + midx * N + nidx, dxval, mask=mmask & nmask)


def triton_fused_loop_bwd(x, w, b, mean, rstd, dy, _y_ignore):
    dx = torch.full_like(x, 0)

    MBLK = 256
    NBLK = 64
    nsplit = cdiv(x.size(0), MBLK)

    interm_w = torch.empty([nsplit, x.shape[1]], device=x.device, dtype=torch.float)
    interm_b = torch.empty([nsplit, x.shape[1]], device=x.device, dtype=torch.float)

    kernel_fused[(nsplit,)](x, w, mean, rstd, dy, dx, interm_w, interm_b,
        x.shape[0], x.shape[1], MBLK, NBLK)

    dw = interm_w.sum(dim=0).to(dtype=w.dtype)
    db = interm_b.sum(dim=0).to(dtype=b.dtype)
    return dx, dw, db
