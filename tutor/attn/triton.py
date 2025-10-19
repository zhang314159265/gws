import triton
import triton.language as tl
import torch
import math

def cdiv(a, b):
    return (a + b - 1) // b

@triton.jit
def attn_fwd_kernel(
    Q, K, V,
    Y, rowmaxptr, rowsumptr,
    S, D: tl.constexpr,
    scale,
    QYBLK: tl.constexpr,
    KVBLK: tl.constexpr,
):
    pid0 = tl.program_id(0)
    pid1 = tl.program_id(1)
    Q += pid1 * S * D
    K += pid1 * S * D
    V += pid1 * S * D
    Y += pid1 * S * D
    rowmaxptr += pid1 * S
    rowsumptr += pid1 * S

    qyoff = pid0 * QYBLK
    qyidx = qyoff + tl.arange(0, QYBLK)[:, None]
    qymask = qyidx < S
    didx = tl.arange(0, D)[None, :]

    Qval = tl.load(Q + qyidx * D + didx, qymask, other=0.0)
    Yval = tl.full([QYBLK, D], 0.0, dtype=tl.float32)
    rowmax = tl.full([QYBLK], float("-inf"), dtype=tl.float32)
    rowsum = tl.full([QYBLK], 0.0, dtype=tl.float32)

    for kvoff in range(0, S, KVBLK):
        kvidx = (kvoff + tl.arange(0, KVBLK))[:, None]
        kvmask = kvidx < S

        # load K/V
        Kval = tl.load(K + kvidx * D + didx, kvmask, other=0.0)
        Vval = tl.load(V + kvidx * D + didx, kvmask, other=0.0)
        
        w = tl.dot(Qval, tl.trans(Kval)).to(tl.float32) * scale

        # mask
        w += tl.where(tl.trans(kvidx) <= qyidx, 0.0, float("-inf"))

        blkmax = tl.max(w, axis=1)

        # update rowmax/rowsum
        newrowmax = tl.where(rowmax > blkmax, rowmax, blkmax)
        rowsum = rowsum * tl.exp(rowmax - newrowmax) + tl.exp(w - newrowmax[:, None]).sum(axis=1)

        Yval *= tl.exp(rowmax - newrowmax)[:, None]
        Yval += tl.dot(tl.exp(w - newrowmax[:, None]).to(tl.float16), Vval)

        # put this last
        rowmax = newrowmax
    Yval /= rowsum[:, None]
    tl.store(Y + qyidx * D + didx, Yval, mask=qymask)
    tl.store(rowmaxptr + qyidx, rowmax[:, None], mask=qymask)
    tl.store(rowsumptr + qyidx, rowsum[:, None], mask=qymask)

def attn_fwd(Q, K, V):
    B, H, S, D = Q.shape
    Y = torch.empty_like(Q)
    rowmax = torch.empty([B, H, S], dtype=torch.float, device=Q.device)
    rowsum = torch.empty([B, H, S], dtype=torch.float, device=Q.device)

    # assumes Q, K, V are contiguous
    # tunable (perf much better than QYBLK=512)
    # cache should plays a big role here for perf. Use ncu to verify
    QYBLK = 512 // 8
    KVBLK = 64
    scale = 1.0 / math.sqrt(D)
    attn_fwd_kernel[(cdiv(S, QYBLK), B * H)](
        Q, K, V, Y, rowmax, rowsum, S, D, scale, QYBLK, KVBLK
    )
    return Y, rowmax, rowsum

@triton.jit
def attn_bwd_kernel_dq(
    Q, K, V, dY,
    dQ, dK, dV,
    dWWsumptr, rowmaxptr, rowsumptr,
    S, D: tl.constexpr, scale,
    QYBLK: tl.constexpr, KVBLK: tl.constexpr,
):
    pid0 = tl.program_id(0)
    pid1 = tl.program_id(1)

    Q += pid1 * S * D
    K += pid1 * S * D
    V += pid1 * S * D
    dQ += pid1 * S * D
    dK += pid1 * S * D
    dV += pid1 * S * D
    dY += pid1 * S * D
    dWWsumptr += pid1 * S
    rowmaxptr += pid1 * S
    rowsumptr += pid1 * S

    qyoff = pid0 * QYBLK
    qyidx = qyoff + tl.arange(0, QYBLK)[:, None]
    qymask = qyidx < S
    didx = tl.arange(0, D)[None, :]
    
    Qval = tl.load(Q + qyidx * D + didx, qymask, other=0.0)
    dYval = tl.load(dY + qyidx * D + didx, qymask, other=0.0)
    rowmax = tl.load(rowmaxptr + qyidx, mask=qymask, other=0.0)
    rowsum = tl.load(rowsumptr + qyidx, mask=qymask, other=0.0)

    dWWaccum = tl.full([QYBLK, KVBLK], 0.0, dtype=tl.float32)
    for kvoff in range(0, S, KVBLK):
        kvidx = (kvoff + tl.arange(0, KVBLK))[:, None]
        kvmask = kvidx < S
        
        Kval = tl.load(K + kvidx * D + didx, kvmask, other=0.0)
        Vval = tl.load(V + kvidx * D + didx, kvmask, other=0.0)

        # compute W block
        W = tl.dot(Qval, tl.trans(Kval)).to(tl.float32) * scale
        W += tl.where(tl.trans(kvidx) <= qyidx, 0.0, float("-inf"))
        W = tl.exp(W - rowmax) / rowsum

        dW = tl.dot(dYval, tl.trans(Vval)).to(tl.float32)
        dWWaccum += dW * W
    dWWsumval = tl.sum(dWWaccum, axis=1)[:, None]
    tl.store(dWWsumptr + qyidx, dWWsumval, mask=qymask)

    dQval = tl.full([QYBLK, D], 0.0, dtype=tl.float32)
    for kvoff in range(0, S, KVBLK):
        kvidx = (kvoff + tl.arange(0, KVBLK))[:, None]
        kvmask = kvidx < S
        
        Kval = tl.load(K + kvidx * D + didx, kvmask, other=0.0)
        Vval = tl.load(V + kvidx * D + didx, kvmask, other=0.0)

        # compute W block
        W = tl.dot(Qval, tl.trans(Kval)).to(tl.float32) * scale
        W += tl.where(tl.trans(kvidx) <= qyidx, 0.0, float("-inf"))
        W = tl.exp(W - rowmax) / rowsum

        dW = tl.dot(dYval, tl.trans(Vval)).to(tl.float32)
        dW = dW * W - W * dWWsumval
        dW *= scale
        dW = dW.to(tl.float16)

        dQval += tl.dot(dW, Kval)
    
    tl.store(dQ + qyidx * D + didx, dQval, mask=qymask)

@triton.jit
def attn_bwd_kernel_dkdv(
    Q, K, V, dY,
    dQ, dK, dV,
    dWWsumptr, rowmaxptr, rowsumptr,
    S, D: tl.constexpr, scale,
    QYBLK: tl.constexpr, KVBLK: tl.constexpr,
):
    pid0 = tl.program_id(0)
    pid1 = tl.program_id(1)

    Q += pid1 * S * D
    K += pid1 * S * D
    V += pid1 * S * D
    dQ += pid1 * S * D
    dK += pid1 * S * D
    dV += pid1 * S * D
    dY += pid1 * S * D
    dWWsumptr += pid1 * S
    rowmaxptr += pid1 * S
    rowsumptr += pid1 * S

    kvoff = pid0 * KVBLK
    kvidx = kvoff + tl.arange(0, KVBLK)[:, None]
    kvmask = kvidx < S
    didx = tl.arange(0, D)[None, :]

    dKval = tl.full([KVBLK, D], 0.0, dtype=tl.float32)
    dVval = tl.full([KVBLK, D], 0.0, dtype=tl.float32)
    Kval = tl.load(K + kvidx * D + didx, kvmask, other=0.0)
    Vval = tl.load(V + kvidx * D + didx, kvmask, other=0.0)

    for qyoff in range(0, S, QYBLK):
        qyidx = (qyoff + tl.arange(0, QYBLK))[:, None]
        qymask = qyidx < S

        Qval = tl.load(Q + qyidx * D + didx, qymask, other=0.0)
        dYval = tl.load(dY + qyidx * D + didx, qymask, other=0.0)
        dWWsumval = tl.load(dWWsumptr + qyidx, mask=qymask, other=0.0)
        rowmax = tl.load(rowmaxptr + qyidx, mask=qymask, other=0.0)  # dim already expanded
        rowsum = tl.load(rowsumptr + qyidx, mask=qymask, other=0.0)

        # compute W block
        W = tl.dot(Qval, tl.trans(Kval)).to(tl.float32) * scale
        W += tl.where(tl.trans(kvidx) <= qyidx, 0.0, float("-inf"))
        W = tl.exp(W - rowmax) / rowsum

        # dv
        dVval += tl.dot(tl.trans(W.to(tl.float16)), dYval)

        # dk
        dW = tl.dot(dYval, tl.trans(Vval)).to(tl.float32)
        dW = dW * W - W * dWWsumval
        dW *= scale
        dW = dW.to(tl.float16)
        dKval += tl.dot(tl.trans(dW), Qval)

    tl.store(dK + kvidx * D + didx, dKval, mask=kvmask)
    tl.store(dV + kvidx * D + didx, dVval, mask=kvmask)


# XXX This version has a lot of recomputations!
def attn_bwd(dY, Q, K, V, _Y_ignore, rowmax, rowsum):
    B, H, S, D = Q.shape
    dQ = torch.empty_like(Q)
    dK = torch.empty_like(K)
    dV = torch.empty_like(V)
    dWWsum = torch.empty([B, H, S], dtype=torch.float, device=Q.device)

    QYBLK = 512 // 8
    KVBLK = 64
    scale = 1.0 / math.sqrt(D)

    attn_bwd_kernel_dq[(cdiv(S, QYBLK), B * H)](
        Q, K, V, dY, dQ, dK, dV, dWWsum, rowmax, rowsum, S, D, scale, QYBLK, KVBLK,
    )

    attn_bwd_kernel_dkdv[(cdiv(S, KVBLK), B * H)](
        Q, K, V, dY, dQ, dK, dV, dWWsum, rowmax, rowsum, S, D, scale, QYBLK, KVBLK,
    )

    del dWWsum
    return dQ, dK, dV
