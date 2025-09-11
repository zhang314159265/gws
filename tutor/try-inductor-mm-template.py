import torch
import torch._inductor.config as inductor_config

inductor_config.max_autotune = True
inductor_config.max_autotune_gemm_backends = "TRITON"

@torch.compile
def f(a, b):
    return a @ b

x = torch.randn(1024, 1024, device="cuda")
f(x, x)
