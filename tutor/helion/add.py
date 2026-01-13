import torch
import helion
import helion.language as hl

@helion.kernel(
    autotune_effort="none"
    # configs=[helion.Config(block_sizes=[16, 32]), ] # helion.Config(block_sizes=[32, 64]),],
)
def helion_add(x, y):
    # TODO broadcasting and type promotion
    z = torch.empty_like(x)
    for tile in hl.tile(z.shape):
        z[tile] = x[tile] + y[tile]

    return z

x, y = [torch.randn(1024, 1024, device="cuda") for _ in range(2)]
ref = x + y
act = helion_add(x, y)
torch.testing.assert_close(ref, act)
print("PASS")
