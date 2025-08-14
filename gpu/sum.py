import torch
import curun

M, N = 1024, 1024 * 1024

x = torch.randn(M, N, device="cuda", dtype=torch.bfloat16)
ref = x.sum(dim=-1)

y = torch.empty(M, device="cuda", dtype=torch.bfloat16)
num_warps = 4
curun.open("sum.cubin").sym("sum_kernel")[M, 32 * num_warps, num_warps * torch.float.itemsize](x, y, M, N)

torch.cuda.synchronize()
assert torch.allclose(ref, y, atol=1e-2, rtol=1e-2), f"ref:\n{ref}\nact:\n{y}\n"
print("bye")
