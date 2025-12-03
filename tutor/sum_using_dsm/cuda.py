import sys
import os
import torch

sys.path.append(os.path.expanduser("~/gws/gpu")) # for curun. TODO: Build a wheel to avoid this hack..
import curun
cukernel = curun.open(os.path.join(os.path.dirname(__file__), "sum.cu")).sym("sum_kernel")

def cuda_sum(x):
    M, N = x.shape
    y = torch.empty(x.shape, device="cuda")
    grid_size = M
    block_size = 256
    num_warps = block_size // 32
    shared_memory = num_warps * torch.float.itemsize
    cukernel[grid_size, block_size, shared_memory](x, y, M, N)
    return y

cukernel_dsm = curun.open(os.path.join(os.path.dirname(__file__), "sum_dsm.cu")).sym("sum_dsm_kernel")

def cuda_sum_dsm(x):
    M, N = x.shape
    y = torch.empty(x.shape, device="cuda")
    cluster_size = 8  # how to use 16 cluster size?
    grid_size = M * cluster_size
    block_size = 512
    num_warps = block_size // 32
    assert (N * torch.float.itemsize) % cluster_size == 0

    shared_memory = N * torch.float.itemsize // cluster_size + num_warps * torch.float.itemsize
    # print(f"shared_memory = {shared_memory / 1000} KB")

    cukernel_dsm[grid_size, block_size, shared_memory](x, y, M, N)
    # torch.cuda.synchronize(); assert False
    return y
