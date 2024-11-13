import torch
import triton
import triton.language as tl

from common import bench

M = 1024 * 1024 * 1024

@triton.jit
def kernel(x, y, numel, BS: tl.constexpr):
    off = tl.program_id(0) * BS + tl.arange(0, BS)
    mask = off < numel

    x_val = tl.load(x + off, mask)
    y_val = tl.where(x_val > 0, x_val, 0)
    tl.store(y + off, y_val, mask)

def launch(x):
    y = torch.empty_like(x)
    BS = 1024
    kernel[(triton.cdiv(x.numel(), BS),)](x, y, x.numel(), BS)
    return y

x = torch.randn(M, device="cuda")

bench(lambda x: x.relu(), launch, (x,))
