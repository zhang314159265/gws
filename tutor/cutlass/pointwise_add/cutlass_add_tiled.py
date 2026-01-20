import curun
import os
import torch

cukernel = curun.open(os.path.join(os.path.dirname(__file__), "cutlass_add_tiled.cu"), use_cutlass=True).sym("cutlass_kernel")

def cutlass_add_tiled(x, y):
    if os.getenv("USE_ZEROS_LIKE") == "1":
        z = torch.zeros_like(x)
    else:
        z = torch.empty_like(x)
    BS = 256
    BLOCK_M, BLOCK_N = 128, 64
    cukernel[
        (x.size(0) // BLOCK_M, x.size(1) // BLOCK_N),
        BS,
    ](x, y, z, x.size(0), x.size(1))
    return z
