import torch
import sys
import os
sys.path.append(os.path.expanduser("~/gws/gpu")) # for curun. TODO: Build a wheel to avoid this hack..

import curun
cukernel = curun.open(os.path.join(os.path.dirname(__file__), "kernel.cu")).sym("layernorm_bwd_kernel")
def layernorm_bwd(dx, dw, db, scratch, x, w, mean, rstd, dy, grid_size, block_size, shared_mem_size):
    BT, C = x.shape
    cukernel[grid_size, block_size, shared_mem_size](dx, dw, db, scratch, dy, x, w, mean, rstd, BT, C)

multi_processor_count = torch.cuda.get_device_properties(0).multi_processor_count

def cuda_bwd(x, w, b, mean, rstd, dy, _y_ignore):
    dx = torch.empty_like(x)
    dw = torch.empty_like(w)
    db = torch.empty_like(b)

    assert x.ndim == 2
    assert x.dtype == torch.bfloat16, "Changing xdtype need change rounded_C computation"
    assert w.dtype == torch.bfloat16, "the kernel only work for bfloat16 weight for now"

    BT, C = x.shape
    grid_size = 2 * multi_processor_count
    block_size = 512

    denom = 32 * 8 # 32 * x128::size
    rounded_C = C if C % denom == 0 else (C - C % denom + denom)
    shared_mem_size = (2 * rounded_C + 2 * (block_size - 32) * 4) * 4

    scratch = torch.empty(32 + 2 * grid_size * C, device=x.device, dtype=torch.float)
    scratch[:1].fill_(0.0)  # TODO llm.c use a cudaMemsetAsync
    layernorm_bwd(dx, dw, db, scratch, x, w, mean, rstd, dy, grid_size, block_size, shared_mem_size)
    return dx, dw, db
