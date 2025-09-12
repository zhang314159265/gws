import torch
import torch._inductor.config as inductor_config

inductor_config.max_autotune = True
inductor_config.max_autotune_gemm_backends = "TRITON"

@torch.compile
def f(a, b, c):
    x = a @ b
    return x * c

x, y, z = [torch.randn(1024, 1024, device="cuda") for _ in range(3)]
f(x, y, z)
