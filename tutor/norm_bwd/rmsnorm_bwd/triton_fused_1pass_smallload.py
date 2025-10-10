"""
Each thread block handle a [split-size, RBLOCK] split of x.
But instead of loading the whole split at once, this kernel load 1 row
or several row at a time.
"""

import torch
import triton
import triton.language as tl

from ..utils import cdiv, next_power_of_two

@triton.jit
def kernel_fused(x, w, rsqrt, dy, dx, interm, M, N, split_size: tl.constexpr, NBLK: tl.constexpr):
    start_row = tl.program_id(0) * split_size
    intermval = tl.full([NBLK], 0, dtype=tl.float32)

    nidx = tl.arange(0, NBLK)
    nmask = nidx < N

    split_size = min(split_size, M - start_row)
    xptr = x + start_row * N
    dyptr = dy + start_row * N
    dxptr = dx + start_row * N
    rsqrtptr = rsqrt + start_row

    wval = tl.load(w + nidx, mask=nmask, other=0.0).to(tl.float32)

    for _ in range(0, split_size):
        # load
        rsqrtval = tl.load(rsqrtptr).to(tl.float32)
        dyval = tl.load(dyptr + nidx, mask=nmask, other=0.0).to(tl.float32)
        xval = tl.load(xptr + nidx, mask=nmask, other=0.0).to(tl.float32)

        # dw
        intermval += xval * rsqrtval * dyval
        
        # dx
        # y = x * rsqrt * w
        sumval = tl.sum(dyval * xval * wval)
        dx_part1 = dyval * rsqrtval * wval
        dx_part2 = sumval * xval / N * rsqrtval * rsqrtval * rsqrtval
        dxval = dx_part1 - dx_part2
        tl.store(dxptr + nidx, dxval, mask=nmask)

        # increment ptr
        rsqrtptr += 1
        dyptr += N
        xptr += N
        dxptr += N

    tl.store(interm + tl.program_id(0) * N + nidx, intermval, mask=nmask)


def triton_fused_1pass_smallload_bwd(x, w, rsqrt, dy, _y_ignore):
    dx = torch.empty_like(x)
    # split_size = 16 # 1.858 tbps
    # split_size = 32 # 1.954 tbps
    split_size = 64 # 1.971 tbps
    # split_size = 128 # 1.889 tbps
    nsplit = cdiv(x.size(0), split_size)
    NBLK = next_power_of_two(x.size(1))

    interm = torch.empty([nsplit, x.shape[1]], device=x.device, dtype=torch.float)

    kernel_fused[(nsplit,)](x, w, rsqrt, dy, dx, interm, x.shape[0], x.shape[1], split_size, NBLK)
    dw = interm.sum(dim=0)

    return dx, dw
