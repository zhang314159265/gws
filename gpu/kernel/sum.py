import torch
import curun
from util import bench, checkclose

M, N = 1024, 1024 * 1024

x = torch.randn(M, N, device="cuda", dtype=torch.bfloat16)

def baseline(x):
    return x.sum(dim=-1)

@torch.compile
def compiled_baseline(x):
    return x.sum(dim=-1)

num_warps = 32
cukernel = curun.open("kernel/sum.cu").sym("sum_kernel")
def sum_with_cuda(x):
    y = torch.empty(M, device="cuda", dtype=torch.bfloat16)
    cukernel[M, 32 * num_warps, num_warps * torch.float.itemsize](x, y, M, N)
    return y

# verify the accuracy
ref = baseline(x)
act1 = compiled_baseline(x)
act2 = sum_with_cuda(x)
torch.cuda.synchronize()

checkclose(ref, act1, tol=1e-2)
checkclose(ref, act2, tol=1e-2)

total_bytes = (M * N + M) * x.itemsize
bench(lambda: baseline(x), "baseline", total_bytes=total_bytes)
bench(lambda: compiled_baseline(x), "compiled", total_bytes=total_bytes)
bench(lambda: sum_with_cuda(x), "cuda_kernel", total_bytes=total_bytes)

print("bye")
