import torch
import triton
import triton.language as tl

from ..utils import cdiv, next_power_of_two

@triton.jit
def kernel_fused(x, w, rsqrt, dy, dx, interm, M, N, MBLK: tl.constexpr, NBLK: tl.constexpr):
    midx = tl.program_id(0) * MBLK + tl.arange(0, MBLK)
    midx = midx[:, None]
    mmask = midx < M

    nidx = tl.arange(0, NBLK)
    nidx = nidx[None, :]
    nmask = nidx < N

    # load
    rsqrtval = tl.load(rsqrt + midx, mask=mmask, other=0.0).to(tl.float32)
    dyval = tl.load(dy + midx * N + nidx, mask=mmask & nmask, other=0.0).to(tl.float32)
    xval = tl.load(x + midx * N + nidx, mask=mmask & nmask, other=0.0).to(tl.float32)
    wval = tl.load(w + nidx, mask=nmask, other=0.0).to(tl.float32)

    # partial dw
    dwval = tl.sum(xval * rsqrtval * dyval, axis=0)[None, :]
    tl.store(interm + tl.program_id(0) * N + nidx, dwval, mask=nmask)

    # dx
    # y = x * rsqrt * w
    sumval = tl.sum(dyval * xval * wval, axis=1)[:, None]
    dx_part1 = dyval * rsqrtval * wval
    dx_part2 = sumval * xval / N * rsqrtval * rsqrtval * rsqrtval
    dxval = dx_part1 - dx_part2
    tl.store(dx + midx * N + nidx, dxval, mask=mmask & nmask)


def triton_fused_1pass_bwd(x, w, rsqrt, dy, _y_ignore):
    dx = torch.empty_like(x)
    MBLK = 16 # 16% faster than inductor
    NBLK = next_power_of_two(x.size(1))
    nsplit = cdiv(x.size(0), MBLK)

    interm = torch.empty([nsplit, x.shape[1]], device=x.device, dtype=torch.float)

    kernel_fused[(nsplit,)](x, w, rsqrt, dy, dx, interm, x.shape[0], x.shape[1], MBLK, NBLK)
    dw = interm.sum(dim=0)

    return dx, dw
