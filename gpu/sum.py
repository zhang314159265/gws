import torch
import curun
from triton.testing import do_bench

M, N = 1024, 1024 * 1024

x = torch.randn(M, N, device="cuda", dtype=torch.bfloat16)

def baseline(x):
    return x.sum(dim=-1)

@torch.compile
def compiled_baseline(x):
    return x.sum(dim=-1)

num_warps = 4
cukernel = curun.open("sum.cubin").sym("sum_kernel")
def sum_with_cuda(x):
    y = torch.empty(M, device="cuda", dtype=torch.bfloat16)
    cukernel[M, 32 * num_warps, num_warps * torch.float.itemsize](x, y, M, N)
    return y

# verify the accuracy
ref = baseline(x)
act1 = compiled_baseline(x)
act2 = sum_with_cuda(x)
torch.cuda.synchronize()

assert torch.allclose(ref, act1, atol=1e-2, rtol=1e-2), f"ref:\n{ref}\nact:\n{act1}\n"
assert torch.allclose(ref, act2, atol=1e-2, rtol=1e-2), f"ref:\n{ref}\nact:\n{act2}\n"

def bench(f, label, total_bytes):
    for _ in range(5):  # warmup
        f()
    ms = do_bench(f)
    gbps = (total_bytes * 1e-9) / (ms * 1e-3)
    print(f"{label}: {ms:.3f}ms {gbps:.3f}gbps")

total_bytes = (M * N + M) * x.itemsize
bench(lambda: baseline(x), "baseline", total_bytes=total_bytes)
bench(lambda: compiled_baseline(x), "compiled", total_bytes=total_bytes)
bench(lambda: sum_with_cuda(x), "cuda_kernel", total_bytes=total_bytes)

print("bye")
