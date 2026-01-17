import torch
import helion
import helion.language as hl
import torch.nn.functional as F

def ref_swiglu(x1, x2):
    return F.silu(x1) * x2

def ref_fwbw(x1, x2):
    out = ref_swiglu(x1, x2)
    out.sum().backward()
    return out

@helion.kernel(autotune_effort="none")
def helion_swiglu_fwd(a, b):
    out = torch.empty_like(a)
    M = out.numel()
    a = a.view(-1)
    b = b.view(-1)
    out_flat = out.view(-1)

    for tile in hl.tile(M):
        a_vals = a[tile].to(torch.float32)
        out_vals = a_vals * torch.sigmoid(a_vals) * b[tile].to(torch.float32)
        out_vals = out_vals.to(a.dtype)
        out_flat[tile] = out_vals
    return out

@helion.kernel(autotune_effort="none")
def helion_swiglu_bwd(dout, x1, x2):
    dx1 = torch.empty_like(x1)
    dx2 = torch.empty_like(x2)

    dout_flat = dout.view(-1)
    x1_flat = x1.view(-1)
    x2_flat = x2.view(-1)
    dx1_flat = dx1.view(-1)
    dx2_flat = dx2.view(-1)

    for tile in hl.tile(x1.numel()):
        x1_vals = x1_flat[tile].to(torch.float32) 
        dout_vals = dout_flat[tile].to(torch.float32)

        # compute dx2
        dx2_vals = x1_vals * torch.sigmoid(x1_vals) * dout_vals
        dx2_flat[tile] = dx2_vals.to(x2.dtype)

        # compute dx1
        x2_vals = x2_flat[tile].to(torch.float32)
        x1_exp = torch.exp(x1_vals)
        x1_exp_plus1 = x1_exp + 1
        dextra = x1_exp / x1_exp_plus1 + x1_vals * x1_exp / x1_exp_plus1 / x1_exp_plus1
        dx1_vals = dout_vals * x2_vals * dextra
        dx1_flat[tile] = dx1_vals.to(x1.dtype)

    return dx1, dx2

class SwigluFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x1, x2):
        out = helion_swiglu_fwd(x1, x2)
        ctx.save_for_backward(x1, x2)
        return out

    @staticmethod
    def backward(ctx, grad_out):
        x1, x2 = ctx.saved_tensors
        dx1, dx2 = helion_swiglu_bwd(grad_out, x1, x2)
        return dx1, dx2

def helion_fwbw(x1, x2):
    out = SwigluFunction.apply(x1, x2)
    out.sum().backward()
    return out
     

B, T, C = 32, 1024, 768
x1, x2 = [torch.randn(B, T, C, device="cuda", dtype=torch.bfloat16, requires_grad=True) for _ in range(2)]

with torch.no_grad():
    x1_clone, x2_clone = x1.clone(), x2.clone()
x1_clone.requires_grad_()
x2_clone.requires_grad_()

ref = ref_fwbw(x1, x2)
act = helion_fwbw(x1_clone, x2_clone)

for a, b in zip([ref, x1.grad, x2.grad], [act, x1_clone.grad, x2_clone.grad]):
    torch.testing.assert_close(a, b)
print("PASS swiglu")
