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
    return Y
