import torch
import helion
import helion.language as hl

@helion.kernel(use_default_config=True)
def helion_exp(x):
    y = torch.empty_like(x)
    for tile in hl.tile(x.shape):
        y[tile] = torch.exp(x[tile])
    return y

x = torch.randn(1024, 1024, device="cuda")
ref = torch.exp(x)
act = helion_exp(x)
torch.testing.assert_close(ref, act)
print("PASS exp")
