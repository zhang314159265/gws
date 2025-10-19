import math
import torch

def attn_fwd(Q, K, V):
    orig_dtype = Q.dtype
    W = Q @ K.transpose(-1, -2)
    W = W.float()

    # scaling
    scale = 1.0 / math.sqrt(Q.size(-1))
    W *= scale

    # masking
    mask = torch.full((W.size(-2), W.size(-1)), True, device=Q.device)
    mask = mask.tril()
    mask = torch.where(mask, 0.0, float("-inf"))
    # W += mask # TODO bring back mask

    # softmax
    W = torch.softmax(W, dim=-1)
    W = W.to(orig_dtype)
    return W @ V

def attn_bwd(dY, Q, K, V, Y):
    # recompute
    # copy from fwd
    if True:
        orig_dtype = Q.dtype
        W = Q @ K.transpose(-1, -2)
        W = W.float()
    
        # scaling
        scale = 1.0 / math.sqrt(Q.size(-1))
        W *= scale
    
        # masking
        mask = torch.full((W.size(-2), W.size(-1)), True, device=Q.device)
        mask = mask.tril()
        mask = torch.where(mask, 0.0, float("-inf"))
        # W += mask # TODO bring this back
    
        # softmax
        W = torch.softmax(W, dim=-1)


    # compute dV
    dV = W.transpose(-1, -2).to(orig_dtype) @ dY

    # compute dW
    dW = dY @ V.transpose(-1, -2)
    dW = dW.float()  # bwd for W.to(orig_dtype) in fwd?

    # bwd for softmax
    dW = dW * W - W * (dW * W).sum(dim=-1, keepdim=True)
    dW *= scale
    dW = dW.to(orig_dtype)

    # compute dQ/dK
    dQ = dW @ K
    dK = dW.transpose(-1, -2) @ Q

    return dQ, dK, dV
