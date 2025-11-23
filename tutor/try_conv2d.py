import torch
from torch import nn

def myconv(x, weight, bias):
    x = x.detach()
    weight = weight.detach()
    bias = bias.detach()

    def _compute_out_dim(d):
        return (d + 2 * padding - (K - 1) - 1) // stride + 1

    Hout = _compute_out_dim(H)
    Wout = _compute_out_dim(W)
    y = torch.zeros(N, Cout, Hout, Wout)
    y += bias[None, :, None, None]

    def _compute_one_plane(kernel, x_plane, y_plane):
        """
        y_plane is inplace updated
        """
        for h in range(Hout):
            for w in range(Wout):
                hin_start = h * stride - padding
                win_start = w * stride - padding

                dp = 0.0 # dot product
                for k1 in range(K):
                    for k2 in range(K):
                        kval = kernel[k1, k2]
                        hin = hin_start + k1
                        win = win_start + k2
                        if hin < 0 or hin >= H or win < 0 or win >= W:
                            xval = 0.0
                        else:
                            xval = x_plane[hin, win]
                        dp += kval * xval
                y_plane[h, w] += dp

    for b in range(N):
        for cout_idx in range(Cout):
            for cin_idx in range(Cin):
                kernel = weight[cout_idx, cin_idx]
                x_plane = x[b, cin_idx]
                y_plane = y[b, cout_idx]

                _compute_one_plane(kernel, x_plane, y_plane)
    return y

def myconv_bwd(x, weight, dy):
    x = x.detach()
    weight = weight.detach()

    dx = torch.zeros_like(x)
    dw = torch.zeros_like(weight)
    db = dy.sum(dim=(0, 2, 3))

    def _compute_one_plane(kernel, x_plane, dx_plane, dw_plane, dy_plane):
        Hout, Wout = dy_plane.shape

        for h in range(Hout):
            for w in range(Wout):
                hin_start = h * stride - padding
                win_start = w * stride - padding

                dyval = dy_plane[h, w]
                for k1 in range(K):
                    for k2 in range(K):
                        kval = kernel[k1, k2]
                        hin = hin_start + k1
                        win = win_start + k2
                        if hin < 0 or hin >= H or win < 0 or win >= W:
                            xval = 0.0
                            validx = False
                        else:
                            xval = x_plane[hin, win]
                            validx = True

                        # dp += kval * xval
                        dxval = dyval * kval
                        dwval = dyval * xval

                        dw_plane[k1, k2] += dwval
                        if validx:
                            dx_plane[hin, win] += dxval

    for b in range(N):
        for cout_idx in range(Cout):
            for cin_idx in range(Cin):
                kernel = weight[cout_idx, cin_idx]
                x_plane = x[b, cin_idx]
                dx_plane = dx[b, cin_idx]
                dw_plane = dw[cout_idx, cin_idx]
                dy_plane = dy[b, cout_idx]

                _compute_one_plane(kernel, x_plane, dx_plane, dw_plane, dy_plane)

    return dx, dw, db

# setup inputs
N, Cin, H, W = 2, 3, 16, 16
K, Cout = 3, 4
stride, padding = 1, 1

conv = nn.Conv2d(Cin, Cout, K, stride=stride, padding=padding)
x = torch.randn(N, Cin, H, W, requires_grad=True)

# compute ref
ref = conv(x)
dy = torch.randn_like(ref)
dref = torch.autograd.grad(
    ref,
    [x, conv.weight, conv.bias],
    dy
)

# call my own implementation
act = myconv(x, conv.weight, conv.bias)
torch.testing.assert_close(ref, act)
dact = myconv_bwd(x, conv.weight, dy)
torch.testing.assert_close(dref, dact, atol=1e-4, rtol=1e-4)

print("Pass")
