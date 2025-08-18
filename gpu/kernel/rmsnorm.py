import torch
from torch import nn
from util import register_custom_op, checkclose, bench
import curun
import functools

checkclose = functools.partial(checkclose, tol=1e-2)

torch.set_grad_enabled(False)
B, T, C = 32, 1024, 4096 # when increasing C, bandwidth starts to drop. Due to cache hit rate drops?
x = torch.randn(B, T, C, device="cuda", dtype=torch.bfloat16)

norm = nn.RMSNorm(C, eps=1e-5).bfloat16().cuda()
norm.weight = nn.Parameter(norm.weight * 2)
weight = norm.weight

def baseline(x):
    return norm(x)

compiled_baseline = torch.compile(baseline)

def rmsnorm_py(x):
    return x * torch.rsqrt(torch.mean(x * x, dim=-1, keepdim=True) + norm.eps) * weight

compiled_rmsnorm_py = torch.compile(rmsnorm_py)

ref = baseline(x)

act = compiled_baseline(x)
checkclose(ref, act)

act = rmsnorm_py(x)
checkclose(ref, act)

act = compiled_rmsnorm_py(x)
checkclose(ref, act)

cukernel = curun.open("kernel/rmsnorm.cu").sym("rmsnorm_kernel")

@register_custom_op("rmsnorm")
def rmsnorm_cu_kernel(x):
    y = torch.empty_like(x)

    # TODO: support passing the eps as an argoument. curun does not
    # accept floating point argument right now.
    num_warps = 16
    cukernel[B * T, 32 * num_warps, num_warps * torch.float.itemsize](x, weight, y, B * T, C)
    return y

@torch.compile
def rmsnorm_cu_kernel_wrapped_in_torch_compile(x):
    return torch.ops.kernel.rmsnorm(x)

act = rmsnorm_cu_kernel(x)
torch.cuda.synchronize();
checkclose(ref, act)

act = rmsnorm_cu_kernel_wrapped_in_torch_compile(x)
checkclose(ref, act);

total_bytes = (B * T * C * 2 + C) * x.itemsize
bench = functools.partial(bench, total_bytes=total_bytes)
bench(lambda: baseline(x), "baseline")
bench(lambda: compiled_baseline(x), "compiled_baseline")
bench(lambda: rmsnorm_py(x), "rmsnorm_py")
bench(lambda: compiled_rmsnorm_py(x), "compiled_rmsnorm_py")
bench(lambda: rmsnorm_cu_kernel(x), "rmsnorm_cu")
bench(lambda: rmsnorm_cu_kernel_wrapped_in_torch_compile(x), "compiled_rmsnorm_cu")

print("bye")
