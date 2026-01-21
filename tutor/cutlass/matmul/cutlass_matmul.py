import os
import torch
import curun

path = os.path.join(os.path.dirname(__file__), "cutlass_matmul.cu")
kernel = curun.open(path, use_cutlass=True).sym("matmul_kernel")

def cutlass_matmul(x, y):
    y = y.t()
    M, K, N = x.size(0), x.size(1), y.size(0)
    assert y.size(1) == K

    bM, bN, bK = 128, 128, 8
    
    z = torch.empty(M, N, device=x.device, dtype=x.dtype)
    if os.getenv("INIT_WITH_ZERO") == "1":
        z.fill_(0.0)

    kernel[
        (M // bM, N // bN),
        (32 * 8,),
    ](
        x, y, z,
        M, N, K,
        x.stride(0), x.stride(1),
        y.stride(0), y.stride(1),
        z.stride(0), z.stride(1),
        bM, bN, bK,
    )

    return z
