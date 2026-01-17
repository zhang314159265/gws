import torch
import helion
import helion.language as hl

@helion.kernel(autotune_effort="none")
def helion_rmsnorm(x, w, eps):
    M, N = x.shape
    assert list(w.shape) == [N]
    y = torch.empty_like(x)

    for tile in hl.tile(M):
        xtile = x[tile, :]
        rsqrt = torch.rsqrt(torch.mean(xtile * xtile, dim=-1, keepdim=True) + eps)
        y[tile, :] = xtile * rsqrt * w[:]
    return y

def torch_rmsnorm(x, w, eps):
    rsqrt = torch.rsqrt(torch.mean(x * x, dim=-1, keepdim=True) + eps)
    return x * rsqrt * w

M, N = 1024, 2048
x = torch.randn(M, N, device="cuda", dtype=torch.bfloat16)
w = torch.randn(N, device="cuda", dtype=torch.bfloat16)
eps = 1e-6
ref = torch_rmsnorm(x, w, eps)
act = helion_rmsnorm(x, w, eps)
torch.testing.assert_close(ref, act)
print("PASS rmsnorm")
