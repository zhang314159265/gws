import triton
import triton.language as tl
from torch._inductor.runtime import triton_helpers
from triton.language.extra import libdevice
import os

EXP_FTZ: tl.constexpr = os.environ.get("EXP_FTZ", "1") == "1"

@triton.jit
def myexp(inp):
    if EXP_FTZ:
        return libdevice.exp2(inp * 1.4426950408889634)
    else:
        return tl.math.exp(inp)

@triton.jit
def online_softmax_reduce(lhs_max, lhs_sum, dim):
    out_max = triton_helpers.max2(lhs_max, dim)
    out_max_keepdim = out_max[:, None]
    delta = tl.where(out_max_keepdim == float("-inf"), 0, lhs_max - out_max_keepdim)
    out_sum = tl.sum(lhs_sum * myexp(delta), dim)
    return out_max, out_sum

@triton.jit
def online_softmax_combine(lhs_max, lhs_sum, rhs_max):
    """
    When we do combine, we assume lhs is the accumulator and rhs is the next
    block of data.
    Then rhs_sum is always 1. With that assumption, we can save some registers
    and computation.
    """
    out_max = triton_helpers.maximum(lhs_max, rhs_max)

    lhs_scale = tl.where(out_max == float("-inf"), 1.0, myexp(lhs_max - out_max))
    rhs_scale = tl.where(out_max == float("-inf"), 1.0, myexp(rhs_max - out_max))

    # Should be
    #   out_sum = lhs_sum * lhs_scale + rhs_sum * rhs_scale
    # but since rhs_sum is all 1, we can simpliy it.
    out_sum = lhs_sum * lhs_scale + rhs_scale
    return out_max, out_sum

@triton.jit
def online_softmax_base_kernel(inp, tmax, tsum, xnumel, rnumel, XBLOCK: tl.constexpr, RBLOCK: tl.constexpr):
    xnumel = 32768
    rnumel = 50257

    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = tl.full([XBLOCK, RBLOCK], True, tl.int1)
    rbase = tl.arange(0, RBLOCK)[None, :]

    accmax = tl.full([XBLOCK, RBLOCK], float("-inf"), tl.float32)
    accsum = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel

        tmp0 = tl.load(inp + (rindex + 50304 * xindex), rmask, eviction_policy="evict_first", other=0.0).to(tl.float32)
        tmp2 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])

        tmp3, tmp4 = online_softmax_combine(accmax, accsum, tmp2)

        accmax = tl.where(rmask, tmp3, accmax)
        accsum = tl.where(rmask, tmp4, accsum)

    tmp5, tmp6 = online_softmax_reduce(accmax, accsum, 1)
    tmp7 = tmp5[:, None]
    tl.store(tmax + (xindex), tmp7, None)

    tmp8 = tmp6[:, None]
    tmp9 = triton_helpers.math.log(tmp8)
    tl.debug_barrier()
    tl.store(tsum + (xindex), tmp9, None)

@triton.jit
def online_softmax_opt_kernel(inp, tmax, tsum, xnumel, rnumel, XBLOCK: tl.constexpr, RBLOCK: tl.constexpr):
    xnumel = 32768
    rnumel = 50257

    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = tl.full([XBLOCK, RBLOCK], True, tl.int1)
    rbase = tl.arange(0, RBLOCK)[None, :]

    accmax = tl.full([XBLOCK, RBLOCK], float("-inf"), tl.float32)
    accsum = tl.full([XBLOCK, RBLOCK], 0, tl.float32)

    rnumel_round = (rnumel // RBLOCK) * RBLOCK
    for roffset in range(0, rnumel_round, RBLOCK):
        rindex = roffset + rbase

        tmp0 = tl.load(inp + (rindex + 50304 * xindex), None, eviction_policy="evict_first").to(tl.float32)
        tmp2 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])

        tmp3, tmp4 = online_softmax_combine(accmax, accsum, tmp2)

        accmax = tmp3
        accsum = tmp4, accsum

    for roffset in range(rnumel_round, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel

        tmp0 = tl.load(inp + (rindex + 50304 * xindex), rmask, eviction_policy="evict_first", other=0.0).to(tl.float32)
        tmp2 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])

        tmp3, tmp4 = online_softmax_combine(accmax, accsum, tmp2)

        accmax = tl.where(rmask, tmp3, accmax)
        accsum = tl.where(rmask, tmp4, accsum)


    tmp5, tmp6 = online_softmax_reduce(accmax, accsum, 1)
    tmp7 = tmp5[:, None]
    tl.store(tmax + (xindex), tmp7, None)

    tmp8 = tmp6[:, None]
    tmp9 = triton_helpers.math.log(tmp8)
    tl.debug_barrier()
    tl.store(tsum + (xindex), tmp9, None)


def call(args, opt=False):
    inp, tmax, tsum = args
    XBLOCK = 1

    if opt:
        RBLOCK = 1024
        num_warps = 4

        online_softmax_opt_kernel[(triton.cdiv(inp.size(0), XBLOCK), )](inp, tmax, tsum, inp.size(0), inp.size(1), XBLOCK, RBLOCK, num_warps=num_warps)
    else:
        RBLOCK = 2048
        num_warps = 16
        online_softmax_base_kernel[(triton.cdiv(inp.size(0), XBLOCK), )](inp, tmax, tsum, inp.size(0), inp.size(1), XBLOCK, RBLOCK, num_warps=num_warps)
