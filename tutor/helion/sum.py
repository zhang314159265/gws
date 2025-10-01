import helion
import helion.language as hl
import torch

@helion.kernel(use_default_config=True)
def helion_sum(x):
    m, n = x.shape
    y = torch.empty(m, device=x.device, dtype=x.dtype)

    for tile in hl.tile(m):
        y[tile] = x[tile, :].sum(dim=-1)
    return y

x = torch.randn(1024, 1024, device="cuda")
ref = torch.sum(x, dim=-1)
act = helion_sum(x)
torch.testing.assert_close(ref, act)
print("PASS sum")
