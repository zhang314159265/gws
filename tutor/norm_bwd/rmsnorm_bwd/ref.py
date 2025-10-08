import torch
import torch.nn.functional as F

def ref_fwd(x, w, eps, *, return_rsqrt=True):
    # return F.rms_norm(x, x.shape[-1:], w, eps)

    orig_dtype = x.dtype
    rstd_dtype = torch.float

    x = x.float()
    rsqrt = torch.rsqrt((x * x).sum(dim=-1) / x.shape[-1] + eps)
    y = (x * rsqrt[:, None] * w).to(dtype=orig_dtype)

    if return_rsqrt:
        return y, rsqrt
    else:
        return y

def ref_bwd(x, w, rsqrt, dy, y):
    x.grad = None
    w.grad = None
    y.backward(dy, retain_graph=True)
    dx = x.grad
    x.grad = None
    dw = w.grad
    w.grad = None
    return dx, dw
