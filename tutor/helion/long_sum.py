import torch
import helion
import helion.language as hl

@helion.kernel(autotune_effort="none")
def helion_long_sum(x):
    m, n = x.shape
    y = torch.empty(m, device=x.device, dtype=x.dtype)

    block_size_n = hl.register_block_size(n)

    for tile_m in hl.tile(m):
        # both works.
        # acc = hl.zeros([tile_m, block_size_n], dtype=torch.float32)
        acc = torch.zeros([tile_m, block_size_n], device=x.device, dtype=torch.float32)
        for tile_n in hl.tile(n, block_size=block_size_n):
            acc += x[tile_m, tile_n]

        y[tile_m] = acc.sum(dim=-1)
    return y

x = torch.randn(1024, 2048, device="cuda", dtype=torch.bfloat16)
ref = torch.sum(x, dim=-1)
act = helion_long_sum(x)
torch.testing.assert_close(ref, act)
print("PASS long_sum")
