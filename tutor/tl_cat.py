import triton
import triton.language as tl
import torch

torch.set_default_device("cuda")

M, N = 1024, 1024
# M, N = 8, 4

tol = {"atol": 1e-3, "rtol": 1e-3}

def test_reordering():
    @triton.jit
    def kernel(x, y, z, rowsize: tl.constexpr):
        inp_idx = tl.program_id(0) * rowsize + tl.arange(0, rowsize)
        xval = tl.load(x + inp_idx)
        yval = tl.load(y + inp_idx)
        outval = tl.cat(xval, yval, can_reorder=True)
    
        out_idx = tl.program_id(0) * rowsize * 2 + tl.arange(0, rowsize * 2)
        tl.store(z + out_idx, outval)
    
    def run_kernel(x, y):
        out = torch.empty(M, N * 2)
        kernel[(M, 1, 1)](x, y, out, N)
        torch.cuda.synchronize()
        return out
    
    x = torch.randn(M, N)
    y = torch.randn(M, N)
    
    ref = torch.cat([x, y], dim=1)
    act = run_kernel(x, y)
    print("tl.cat may reorder elements. So torch.allclose should return false here")
    assert not torch.allclose(ref, act, **tol), f"ref\n{ref}\nact\n{act}"

    print("However rowsum should match")
    ref2 = ref.sum(dim=1)
    act2 = act.sum(dim=1)
    assert torch.allclose(ref2, act2, **tol), f"ref\n{ref2}\nact\n{act2}"
    print("test_reordering bye")

def test_reordering_sum():
    @triton.jit
    def kernel(x, y, z, rowsize: tl.constexpr):
        inp_idx = tl.program_id(0) * rowsize + tl.arange(0, rowsize)
        xval = tl.load(x + inp_idx)
        yval = tl.load(y + inp_idx)
        outval = tl.sum(tl.cat(xval, yval, can_reorder=True), 0)
    
        out_idx = tl.program_id(0)
        tl.store(z + out_idx, outval)
    
    def run_kernel(x, y):
        out = torch.empty(M)
        kernel[(M, 1, 1)](x, y, out, N)
        torch.cuda.synchronize()
        return out
    
    x = torch.randn(M, N)
    y = torch.randn(M, N)
    
    ref = torch.cat([x, y], dim=1).sum(dim=-1)
    act = run_kernel(x, y)
    print("tl.cat followed by a reduction is safe")
    assert torch.allclose(ref, act, **tol), f"ref\n{ref}\nact\n{act}"
    print("test_reordering_sum bye")


test_reordering()
test_reordering_sum()
