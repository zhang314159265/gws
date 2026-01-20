import torch
import os
import curun

cufile = os.path.join(os.path.dirname(__file__), "cutlass_sum.cu")
cukernel = curun.open(cufile, use_cutlass=True).sym("cutlass_kernel")

def cutlass_sum(x):
    M, N = x.shape

    y = torch.empty(M, device=x.device, dtype=x.dtype)
    if os.getenv("INIT_WITH_ZERO") == "1":
        y.fill_(0.0)

    BLK = 256
    cukernel[
        M,
        BLK,
        BLK * torch.float.itemsize,
    ](x, y, M, N)
    return y
