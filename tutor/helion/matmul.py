import torch
import helion
import helion.language as hl

@helion.kernel(autotune_effort="none")
def helion_matmul(x, y):
    z = torch.empty(M, N, device=x.device, dtype=x.dtype)
    for tile_m, tile_n in hl.tile(z.shape):
        acc = hl.zeros([tile_m, tile_n], dtype=torch.float32)
        for tile_k in hl.tile(K):
            acc = torch.addmm(acc, x[tile_m, tile_k], y[tile_k, tile_n])
        z[tile_m, tile_n] = acc  # is there an automatic downcast?
    return z

M, K, N = 512, 1024, 2048
x = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)
y = torch.randn(K, N, device="cuda", dtype=torch.bfloat16)
ref = x @ y
act = helion_matmul(x, y)
torch.testing.assert_close(ref, act)
print("PASS")
