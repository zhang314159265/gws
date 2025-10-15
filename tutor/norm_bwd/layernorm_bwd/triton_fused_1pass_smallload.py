import torch
import triton
import triton.language as tl

from ..utils import cdiv, next_power_of_two

@triton.jit
def kernel_fused(x, w, mean, rstd, dy, dx, interm_w, interm_b, M, N, split_size: tl.constexpr, NBLK: tl.constexpr):
    start_row = tl.program_id(0) * split_size
    accum_dw = tl.full([NBLK], 0, dtype=tl.float32)
    accum_db = tl.full([NBLK], 0, dtype=tl.float32)

    nidx = tl.arange(0, NBLK)
    nmask = nidx < N

    split_size = min(split_size, M - start_row)
    xptr = x + start_row * N  
    dyptr = dy + start_row * N
    dxptr = dx + start_row * N
    rstdptr = rstd + start_row
    meanptr = mean + start_row

    wval = tl.load(w + nidx, mask=nmask, other=0.0).to(tl.float32)

    for _ in range(0, split_size):
        # load
        rstdval = tl.load(rstdptr).to(tl.float32)
        meanval = tl.load(meanptr).to(tl.float32)
        dyval = tl.load(dyptr + nidx, mask=nmask, other=0.0).to(tl.float32)
        xval = tl.load(xptr + nidx, mask=nmask, other=0.0).to(tl.float32)

        norm = (xval - meanval) * rstdval
    
        # dx
        gnorm = dyval * wval
        mean_gnorm = tl.sum(gnorm) / N
        mean_gnorm_norm = tl.sum(gnorm * norm) / N
    
        dxval = gnorm - mean_gnorm - mean_gnorm_norm * norm
        dxval *= rstdval
        tl.store(dxptr + nidx, dxval, mask=nmask)

        # increment ptr
        rstdptr += 1
        meanptr += 1
        dyptr += N
        xptr += N
        dxptr += N

        # partial dw/db
        accum_dw += dyval * norm
        accum_db += dyval

    tl.store(interm_w + tl.program_id(0) * N + nidx, accum_dw, mask=nmask)
    tl.store(interm_b + tl.program_id(0) * N + nidx, accum_db, mask=nmask)

def triton_fused_1pass_smallload_bwd(x, w, b, mean, rstd, dy, _y_ignore):
    dx = torch.empty_like(x)

    # split_size = 32 # 1.709 tbps
    # split_size = 64 # 1.784 tbps
    split_size = 128 # 1.807 tbps
    # split_size = 256 # 1.679 tbps

    nsplit = cdiv(x.size(0), split_size)
    NBLK = next_power_of_two(x.size(1))

    interm_w = torch.empty([nsplit, x.shape[1]], device=x.device, dtype=torch.float)
    interm_b = torch.empty([nsplit, x.shape[1]], device=x.device, dtype=torch.float)

    kernel_fused[(nsplit,)](x, w, mean, rstd, dy, dx, interm_w, interm_b,
        x.shape[0], x.shape[1], split_size, NBLK)

    dw = interm_w.sum(dim=0).to(dtype=w.dtype)
    db = interm_b.sum(dim=0).to(dtype=b.dtype)
    return dx, dw, db
