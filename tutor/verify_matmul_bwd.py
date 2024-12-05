import torch

M = 32768
K = 768
N = 65536

dtype = torch.bfloat16
A = torch.randn(M, K, device="cuda", requires_grad=True, dtype=dtype)
B = torch.randn(K, N, device="cuda", requires_grad=True, dtype=dtype)

out = A @ B
dout = torch.randn(M, N, device="cuda", dtype=dtype)
out.backward(dout)

ref_dA = A.grad
ref_dB = B.grad

dA = dout @ B.t()
dB = A.t() @ dout

tol = {}
assert torch.allclose(ref_dA, dA, **tol)
assert torch.allclose(ref_dB, dB, **tol)

print("bye")
