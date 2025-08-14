import torch
from torch import nn
from util import register_custom_op
import curun

torch.set_grad_enabled(False)

B, T, C = 32, 1024, 4096
x = torch.randn(B, T, C, device="cuda", dtype=torch.bfloat16)
y = torch.empty_like(x)

curun.open("rmsnorm.cubin").sym("rmsnorm_kernel")[B * T, 32 * 4](x, y, B * T, C)
torch.cuda.synchronize()
assert False, "hlt"

norm = nn.RMSNorm(C, eps=1e-5).bfloat16().cuda()
norm.weight = nn.Parameter(norm.weight * 2)

@register_custom_op("rmsnorm_py")
def rmsnorm_py_kernel(x):
    w = norm.weight
    return x * torch.rsqrt(torch.mean(x * x, dim=-1, keepdim=True) + norm.eps) * w

@register_custom_op("rmsnorm_cu")
def rmsnorm_cu_kernel(x):
    y = torch.empty_like(x)
    curun.open("rmsnorm.cubin").sym("rmsnorm_kernel")[B * T, 32 * 4](x, y, B * T, C)
    return y

PICK = 2

@torch.compile
def f(x):
    if PICK == 0:
        return norm(x)
    elif PICK == 1:
        return torch.ops.kernel.rmsnorm_py(x)
    else:
        return torch.ops.kernel.rmsnorm_cu(x)

ref = norm(x)
act = f(x)
torch.cuda.synchronize()
assert torch.allclose(ref, act, atol=1e-2, rtol=1e-2), f"ref:\n{ref}\nact:\n{act}\n"
print("bye")
