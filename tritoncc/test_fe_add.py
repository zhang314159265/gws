import triton
import triton.language as tl
import torch
import tritoncc.frontend as fe

def fn(a_ptr, b_ptr, c_ptr, num, BLOCK_SIZE: tl.constexpr):
    idx = tl.arange(0, BLOCK_SIZE) + tl.program_id(0) * BLOCK_SIZE
    mask = idx < num
    a = tl.load(a_ptr + idx, mask=mask)
    b = tl.load(b_ptr + idx, mask=mask)
    c = a + b
    tl.store(c_ptr + idx, c, mask=mask)

torch.set_default_device("cuda")
a = torch.rand(1024)
b = torch.rand(1024)
act = torch.empty(1024)
BLOCK_SIZE = 32
fe.compile(fn)(a, b, act, a.numel(), BLOCK_SIZE=BLOCK_SIZE, gridDim=32, blockDim=32 * 4, shared=0)

ref = a + b
tol = 1e-3
assert torch.allclose(ref, act, atol=tol, rtol=tol)
print("PASS")
