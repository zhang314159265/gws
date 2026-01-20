import torch
import curun
import os

path = os.path.join(os.path.dirname(__file__), "cutlass_basic.cu")
kernel = curun.open(path, use_cutlass=True).sym("cutlass_kernel")

def cutlass_basic(x):
    y = torch.empty_like(x)
    if os.environ.get("INIT_WITH_ZERO") == "1":
        y.fill_(0.0)

    M, N = x.shape
    BLKM, BLKN = 128, 64
    kernel[
        (M // BLKM, N // BLKN),
        256,
    ](x, y, M, N)
    return y
