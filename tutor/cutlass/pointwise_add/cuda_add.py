import curun
import os
import torch

cukernel = curun.open(os.path.join(os.path.dirname(__file__), "cuda_add.cu")).sym("cuda_add_kernel")

def cuda_add(x, y):
    z = torch.empty_like(x)
    BS = 256
    GRID = (x.numel() + BS - 1) // BS
    cukernel[
        GRID,
        BS,
    ](x, y, z, x.numel())
    return z
