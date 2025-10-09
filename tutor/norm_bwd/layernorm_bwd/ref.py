import torch
import torch.nn.functional as F

def ref_fwd(x, w, b, eps, *, return_mean_rstd=True, check=False):
    orig_dtype = x.dtype

    y = x.float()
    var, mean = torch.var_mean(y, dim=-1, keepdim=True, correction=0)
    rstd = torch.rsqrt(var + eps)
    y = ((y - mean) * rstd) * w + b
    y = y.to(orig_dtype)

    if check:
        torch.testing.assert_close(y, F.layer_norm(x, x.shape[-1:], w, b, eps))
    if return_mean_rstd:
        return y, mean, rstd
    else:
        return y

def ref_bwd(x, w, b, mean, rstd, dy, y):
    x.grad = None
    w.grad = None
    b.grad = None

    y.backward(dy, retain_graph=True)
    dx = x.grad
    x.grad = None

    dw = w.grad
    w.grad = None

    db = b.grad
    b.grad = None

    return dx, dw, db
