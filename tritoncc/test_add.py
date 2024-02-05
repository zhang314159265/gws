"""
Generated ttir before and after make_ttir: https://gist.github.com/shunting314/f2b578bc7da71459b67bebb2836dfdda
"""
import triton
import triton.language as tl
import torch

def fn(a_ptr, b_ptr, c_ptr, num, BLOCK_SIZE: tl.constexpr):
    idx = tl.arange(0, BLOCK_SIZE) + tl.program_id(0) * BLOCK_SIZE
    mask = idx < num
    a = tl.load(a_ptr + idx, mask=mask)
    b = tl.load(b_ptr + idx, mask=mask)
    c = a + b
    tl.store(c_ptr + idx, c, mask=mask)

jit_fn = triton.jit(fn)

torch.set_default_device("cuda")
a = torch.rand(1024)
b = torch.rand(1024)
act = torch.empty(1024)

ref = a + b
BLOCK_SIZE = 32
grid = ((a.numel() + BLOCK_SIZE - 1) // BLOCK_SIZE,)
compiled = jit_fn.run(a, b, act, a.numel(), BLOCK_SIZE=BLOCK_SIZE, grid=grid, warmup=False)
assert torch.allclose(ref, act)
print("bye poi")
