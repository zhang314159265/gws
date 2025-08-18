import torch
from torch import nn
from util import register_custom_op, checkclose, bench, check_and_bench
import curun
import functools
from torch._inductor import config as inductor_config

inductor_config.benchmark_kernel = False

torch.set_grad_enabled(False)
B, T, C = 32, 1024, 4096 # when increasing C, bandwidth starts to drop. Due to cache hit rate drops?
x = torch.randn(B, T, C, device="cuda", dtype=torch.bfloat16)
total_bytes = (B * T * C * 2 + C) * x.itemsize

norm = nn.RMSNorm(C, eps=1e-5).bfloat16().cuda()
norm.weight = nn.Parameter(norm.weight * 2)
weight = norm.weight

def baseline(x):
    return norm(x)
ref = baseline(x)

check_and_bench = functools.partial(check_and_bench, args=(x,), ref=ref, total_bytes=total_bytes, tol=1e-2)

compiled_baseline = torch.compile(baseline)

def rmsnorm_py(x):
    return x * torch.rsqrt(torch.mean(x * x, dim=-1, keepdim=True) + norm.eps) * weight

compiled_rmsnorm_py = torch.compile(rmsnorm_py)

cukernel = curun.open("kernel/rmsnorm.cu").sym("rmsnorm_kernel")

def run_kernel(kernel, x, num_warps=16):
    y = torch.empty_like(x)

    # TODO: support passing the eps as an argoument. curun does not
    # accept floating point argument right now.
    kernel[B * T, 32 * num_warps, num_warps * torch.float.itemsize](x, weight, y, C)
    return y


@register_custom_op("rmsnorm")
def rmsnorm_cu_kernel(x):
    return run_kernel(cukernel, x)

@torch.compile
def rmsnorm_cu_kernel_wrapped_in_torch_compile(x):
    return torch.ops.kernel.rmsnorm(x)

manualptxcukernel = curun.open("kernel/rmsnorm_from_cuda.ptx").sym("rmsnorm_kernel_from_cuda_ptx")
def rmsnorm_manual_ptx_kernel(x):
    return run_kernel(manualptxcukernel, x)

tritonptxcukernel = curun.open("kernel/rmsnorm_triton.ptx").sym("rmsnorm_kernel_triton_ptx")
def rmsnorm_triton_ptx_kernel(x):
    return run_kernel(tritonptxcukernel, x)

tritonptxv4cukernel = curun.open("kernel/rmsnorm_triton_v4.ptx").sym("rmsnorm_kernel_triton_ptx_v4")
def rmsnorm_triton_ptx_v4_kernel(x):
    return run_kernel(tritonptxv4cukernel, x)


bench(lambda: baseline(x), "baseline", total_bytes=total_bytes)
check_and_bench(compiled_baseline, label="compiled_baseline")
check_and_bench(rmsnorm_py)
check_and_bench(compiled_rmsnorm_py, label="compiled_rmsnorm_py")
check_and_bench(rmsnorm_cu_kernel)
check_and_bench(rmsnorm_cu_kernel_wrapped_in_torch_compile)
check_and_bench(rmsnorm_manual_ptx_kernel)
check_and_bench(rmsnorm_triton_ptx_kernel)
check_and_bench(rmsnorm_triton_ptx_v4_kernel) # does not help much!

print("bye")
