import sys
import os
import torch

sys.path.append(os.path.expanduser("~/gws/gpu")) # for curun. TODO: Build a wheel to avoid this hack..
import curun
cukernel = curun.open(os.path.join(os.path.dirname(__file__), "sum.cu")).sym("sum_kernel")

def cuda_sum(x, y=None):
    M, N = x.shape
    if y is None:
        y = torch.empty_like(x)
    grid_size = M
    block_size = 256
    num_warps = block_size // 32
    shared_memory = num_warps * torch.float.itemsize
    cukernel[grid_size, block_size, shared_memory](x, y, M, N, x.dtype == torch.float32)
    return y

cukernel_dsm = curun.open(os.path.join(os.path.dirname(__file__), "sum_dsm.cu")).sym("sum_dsm_kernel")

def cuda_sum_dsm(x, y=None):
    M, N = x.shape
    if y is None:
        y = torch.empty_like(x)
    cluster_size = 8  # how to use 16 cluster size?
    grid_size = M * cluster_size
    block_size = 512
    num_warps = block_size // 32
    assert (N * x.itemsize) % cluster_size == 0

    if x.dtype == torch.bfloat16:
        block_size = 256

    shared_memory = N * x.itemsize // cluster_size + num_warps * torch.float.itemsize
    # print(f"shared_memory = {shared_memory / 1000} KB")

    cukernel_dsm[grid_size, block_size, shared_memory](x, y, M, N, x.dtype == torch.float32)
    # torch.cuda.synchronize(); assert False
    return y
