import curun
import os
import torch

cukernel = curun.open(os.path.join(os.path.dirname(__file__), "cutlass_add_basic.cu"), use_cutlass=True).sym("cutlass_kernel")

def cutlass_add_basic(x, y):
    z = torch.empty_like(x)
    BS = 256
    GRID = (x.numel() + BS - 1) // BS
    cukernel[
        GRID,
        BS,
    ](x, y, z, x.numel())
    return z
