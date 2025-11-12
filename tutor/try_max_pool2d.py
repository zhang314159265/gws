"""
TODO:
2. backward with indices
"""

import torch
import torch.nn.functional as F

def padlist(x):
    if isinstance(x, (list, tuple)):
        return x
    else:
        return [x, x]

def compute_newhw(oldhw, kernel_size, stride, padding):
    newhw = []
    assert len(oldhw) == 2
    for idx in range(2):
        newd = (oldhw[idx] + 2 * padding[idx] - 1 - (kernel_size[idx] - 1) // stride[idx]) + 1
        newhw.append(newd)
    return newhw

def my_max_pool2d(x, kernel_size, stride, padding, dilation, ceil_mode, return_indices):
    assert dilation == 1
    assert not ceil_mode

    kernel_size, stride, padding = (padlist(v) for v in (kernel_size, stride, padding))

    N, C, oldH, oldW = x.shape
    newH, newW = compute_newhw([oldH, oldW], kernel_size, stride, padding)

    y = torch.empty(N, C, newH, newW, device=x.device, dtype=x.dtype)
    indices = None
    if return_indices:
        indices = torch.zeros(N, C, newH, newW, device=x.device, dtype=torch.int64)

    NINF = -1e20
    for n in range(N):
        for c in range(C):
            for h in range(newH):
                for w in range(newW):
                    # compute item
                    starth = h * stride[0] - padding[0]
                    startw = w * stride[1] - padding[1]

                    val = NINF
                    ind = -1
                    for k1 in range(kernel_size[0]):
                        for k2 in range(kernel_size[1]):
                            idx1 = starth + k1
                            idx2 = startw + k2

                            inval = x[n][c][idx1][idx2] if (idx1 >= 0 and idx1 < oldH and idx2 >= 0 and idx2 < oldW) else NINF
                            if inval > val:
                                ind = idx1 * oldW + idx2
                            val = max(val, inval)
                    y[n][c][h][w] = val
                    indices[n][c][h][w] = ind

    return y, indices if return_indices else y

def my_max_pool2d_bwd(x, y, indices, dy):
    dx = torch.zeros_like(x)
    N, C, H, W = x.shape
    idx0 = indices // W
    idx1 = indices % W
    
    # TODO: how to use a pytorch op to do this
    for n in range(N):
        for c in range(C):
            for h in range(H):
                for w in range(W):
                    dyval = dy[n][c][h][w]
                    x = idx0[n][c][h][w]
                    y = idx1[n][c][h][w]
                    dx[n][c][x][y] += dyval
    return dx

N, C, H, W = 2, 3, 16, 16
x = torch.randn(N, C, H, W, device="cpu", requires_grad=True)

args = (x, )
kwargs = dict(kernel_size=3, stride=1, padding=1, dilation=1, ceil_mode=False, return_indices=True)
ref = F.max_pool2d(*args, **kwargs)
dy = torch.randn_like(ref[0])
ref_dx = torch.autograd.grad(ref[0], x, dy)[0]

act = my_max_pool2d(*args, **kwargs)
torch.testing.assert_close(ref, act)

act_dx = my_max_pool2d_bwd(x, *act, dy)
torch.testing.assert_close(ref_dx, act_dx)
print("bye")
