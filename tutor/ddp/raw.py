from .utils import printall, print0
import torch.distributed as dist
from torch.distributed.distributed_c10d import _get_default_group
import torch
import torch.nn.functional as F
from .train_step import train_step

# TODO sync buffers
def sync_parameters(model, coalesce=True):
    params = []
    for param in model.parameters():
        params.append(param.detach())
    if not coalesce:
        for param in params:
            dist.broadcast(param, src=0)
    else:
        # bucketize the broadcasting to be more efficient for
        # many small tensors.
        pg = _get_default_group()
        buffer_size = 25 * 1024 * 1024
        dist._broadcast_coalesced(pg, params, buffer_size, src=0)

def sync_grads(model):
    for param in model.parameters():
        grad = param.grad
        dist.all_reduce(grad, op=dist.ReduceOp.AVG)

# initial hash dab0fb1a
def raw_ddp(model, datagen, optim):
    sync_parameters(model)
    # printall(f"Initial hash: {model.param_hash()}")

    for stepno in range(2):
        def _all_reduce_fn():
            sync_grads(model)
        train_step(model, datagen, optim, inject_all_reduce=_all_reduce_fn)
