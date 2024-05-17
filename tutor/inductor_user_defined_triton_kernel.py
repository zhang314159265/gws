"""
Here is the user defined triton kernel extracted from the wrapper with some
benchmarking code: https://gist.github.com/shunting314/443cab1aa2539c17d5ddfcbf82bea37b

Inductor right now explicitly skips user defined triton kernel for coordinated descent tuning.

The whole wrapper: https://gist.github.com/shunting314/9201ce28169f9b62c316f75732b6b539

I think inductor can not fuse it's generated kernel with user defined triton kernel as well.
Since the ReLU (inductor generate) is not fused the with Add kernel (user defined).
"""

import torch
import triton
import triton.language as tl
import torch.nn.functional as F
from torch._inductor import config as inductor_config

inductor_config.benchmark_kernel = True
inductor_config.triton.unique_kernel_names = True
inductor_config.coordinate_descent_tuning = True

torch.set_default_device("cuda")

@triton.jit
def add_kernel(x_ptr, y_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    output = x + y
    tl.store(output_ptr + offsets, output, mask=mask)

@torch.compile
def f(a, b):
    c = torch.empty_like(a)
    BLOCK_SIZE = 512
    add_kernel[(triton.cdiv(a.numel(), BLOCK_SIZE),)](a, b, c, a.numel(), BLOCK_SIZE=BLOCK_SIZE)
    return F.relu(c)

a = torch.randn(1024 * 1024)
b = torch.randn(1024 * 1024)
actual = f(a, b)
expected = F.relu(a + b)
assert torch.allclose(actual, expected)
print("bye")
