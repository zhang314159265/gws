import torch
import triton
import triton.language as tl

def _alloc(size, alignment, stream):
    return torch.empty(size, device="cuda", dtype=torch.int8)

triton.set_allocator(_alloc)

@triton.jit
def kernel(x, y, z, M, N, XBLK: tl.constexpr, YBLK: tl.constexpr):
    pid0 = tl.program_id(0)
    pid1 = tl.program_id(1)

    xdesc = tl.make_tensor_descriptor(x, [M, N], [N, 1], [XBLK, YBLK])
    ydesc = tl.make_tensor_descriptor(y, [M, N], [N, 1], [XBLK, YBLK])
    zdesc = tl.make_tensor_descriptor(z, [M, N], [N, 1], [XBLK, YBLK])

    xval = xdesc.load([pid0 * XBLK, pid1 * YBLK])  # offset is in element rather than block
    yval = ydesc.load([pid0 * XBLK, pid1 * YBLK])
    zval = xval + yval

    # zdesc.store([pid0 * XBLK, pid1 * YBLK], zval)
    # we can also mix tensor descriptor with raw pointer
    tl.store(z + (pid0 * XBLK + tl.arange(0, XBLK))[:, None] * N + (pid1 * YBLK + tl.arange(0, YBLK))[None, :], zval);

x, y = [torch.randn(1024, 1024, device="cuda") for _ in range(2)]
ref = x + y
act = torch.empty_like(x)

XBLK, YBLK = 32, 32
kernel[x.size(0) // XBLK, x.size(1) // YBLK](x, y, act, x.size(0), x.size(1), XBLK, YBLK)

torch.testing.assert_close(ref, act)
print("PASS")
