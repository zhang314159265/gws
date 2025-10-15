import torch
import triton
import triton.language as tl

from ..utils import cdiv, next_power_of_two

@triton.jit
def kernel_fused(x, w, mean, rstd, dy, dx, interm_w, interm_b, M, N, MBLK: tl.constexpr, NBLK: tl.constexpr):
    midx = tl.program_id(0) * MBLK + tl.arange(0, MBLK)
    midx = midx[:, None]
    mmask = midx < M

    nidx = tl.arange(0, NBLK)
    nidx = nidx[None, :]
    nmask = nidx < N

    # load
    rstdval = tl.load(rstd + midx, mask=mmask, other=0.0).to(tl.float32)
    meanval = tl.load(mean + midx, mask=mmask, other=0.0).to(tl.float32)
    dyval = tl.load(dy + midx * N + nidx, mask=mmask & nmask, other=0.0).to(tl.float32)
    xval = tl.load(x + midx * N + nidx, mask=mmask & nmask, other=0.0).to(tl.float32)
    wval = tl.load(w + nidx, mask=nmask, other=0.0).to(tl.float32)

    norm = (xval - meanval) * rstdval

    # partial dw
    dwval = tl.sum(dyval * norm, axis=0)[None, :]
    tl.store(interm_w + tl.program_id(0) * N + nidx, dwval, mask=nmask)
    dbval = tl.sum(dyval, axis=0)[None, :]
    tl.store(interm_b + tl.program_id(0) * N + nidx, dbval, mask=nmask)

    # dx
    gnorm = dyval * wval
    mean_gnorm = tl.sum(gnorm, axis=1)[:, None] / N
    mean_gnorm_norm = tl.sum(gnorm * norm, axis=1)[:, None] / N

    dxval = gnorm - mean_gnorm - mean_gnorm_norm * norm
    dxval *= rstdval
    tl.store(dx + midx * N + nidx, dxval, mask=mmask & nmask)


def triton_fused_1pass_bwd(x, w, b, mean, rstd, dy, _y_ignore):
    dx = torch.empty_like(x)

    MBLK = 16 # 27% faster than inductor
    # MBLK = 8 # 1.146 tbgs
    # MBLK = 32 # 0.640 tbgs
    NBLK = next_power_of_two(x.size(1))
    nsplit = cdiv(x.size(0), MBLK)

    interm_w = torch.empty([nsplit, x.shape[1]], device=x.device, dtype=torch.float)
    interm_b = torch.empty([nsplit, x.shape[1]], device=x.device, dtype=torch.float)

    kernel_fused[(nsplit,)](x, w, mean, rstd, dy, dx, interm_w, interm_b,
        x.shape[0], x.shape[1], MBLK, NBLK)

    dw = interm_w.sum(dim=0).to(dtype=w.dtype)
    db = interm_b.sum(dim=0).to(dtype=b.dtype)
    return dx, dw, db
