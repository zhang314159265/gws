import torch
import helion
import helion.language as hl

@helion.kernel(config=helion.Config(block_sizes=[64]), torch_compile_fusion=True)
def kernel(a):
    b = torch.empty_like(a)
    for tile in hl.tile(a.shape):
        b[tile] = a[tile] + 1
    return b

@torch.compile
def f(a):
    a = kernel(a)
    return a.sum()

a = torch.randn(1024, device="cuda")
actual = f(a)
expected = (a + 1).sum()
torch.testing.assert_close(expected, actual)
