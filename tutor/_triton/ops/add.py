import torch
import triton.language as tl
import triton

from common import bench

@triton.jit
def kernel(x, y, z, numel, BLOCK_SIZE: tl.constexpr):
    off = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = off < numel

    x_val = tl.load(x + off, mask=mask)
    y_val = tl.load(y + off, mask=mask)
    z_val = x_val + y_val
    tl.store(z + off, z_val, mask=mask)

def launch(x, y):
    z = torch.empty_like(x)
    BLOCK_SIZE = 1024
    kernel[((x.numel() + BLOCK_SIZE - 1) // BLOCK_SIZE,)](x, y, z, x.numel(), BLOCK_SIZE=BLOCK_SIZE)
    return z

M = 1024 * 1024
N = 1024
x = torch.randn(M, N, device="cuda")
y = torch.randn(M, N, device="cuda")

bench(lambda x, y: x + y, launch, (x, y))
