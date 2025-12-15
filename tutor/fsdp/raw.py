"""
TODO:
- this version assumes all tensors are float32. Mix precision does not work so far.
"""

from .util import compute_tensor_hash
from ..ddp.utils import print0, printall
import torch.distributed as dist
import torch
from torch import nn
import torch.nn.functional as F
from torch.optim import SGD

def get_flat_param(model):
    tlist = []
    numels = []
    for p in model.parameters():
        tlist.append(p.flatten())
        numels.append(p.numel())
    flat = torch.concat(tlist)
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    assert flat.numel() % world_size == 0
    size = flat.numel() // world_size
    out = flat[rank * size : (rank + 1) * size]
    # XXX do a clone so we don't be a view to the underlying tensor.
    # Is there an equivalanet operation done in PyTorch FSDP?
    out = nn.Parameter(out.detach().clone())
    return out, numels

def get_flat_grad(model):
    tlist = []
    for p in model.parameters():
        tlist.append(p.grad.flatten())
    flat = torch.concat(tlist)
    return flat.detach()

def setup_params_as_view(model, whole_tensor, numels):
    accum = 0
    for param, numel in zip(model.parameters(), numels):
        flat_view = whole_tensor[accum: accum + numel]
        data = flat_view.view(param.shape)
        param.data = data
        accum += numel

def raw_fsdp(model, datagen, lr):
    # setup flat param
    flat_param, numels = get_flat_param(model)
    printall(f"flat param hash before training {compute_tensor_hash(flat_param)}")

    # XXX: how does PyTorch FSDP release the weight tensors.
    optim = SGD([flat_param], lr=lr)

    rank = dist.get_rank()
    world_size = dist.get_world_size()
    for step in range(2):
        x, label = datagen.generate(32)
        # printall(f"hash of x {compute_tensor_hash(x)} label {compute_tensor_hash(label)}")

        # fwd allgather
        # printall(f"flat param before fwd allgather {compute_tensor_hash(flat_param.data)}")
        whole_tensor = torch.empty(flat_param.numel() * world_size, device=flat_param.device, dtype=flat_param.dtype)
        dist.all_gather_into_tensor(whole_tensor, flat_param.data)
        setup_params_as_view(model, whole_tensor.detach(), numels)
        whole_tensor = None # release

        # XXX but the parameters still refer to the whole_tensor?
        # How can I release the parameter while still keeping the shape?

        probs = model(x)
        loss = F.binary_cross_entropy(probs.flatten(), label)

        # bwd allgather # TODO dedup with fwd
        # printall(f"flat param before bwd allgather {compute_tensor_hash(flat_param.data)}")
        whole_tensor = torch.empty(flat_param.numel() * world_size, device=flat_param.device, dtype=flat_param.dtype)
        dist.all_gather_into_tensor(whole_tensor, flat_param.data)
        setup_params_as_view(model, whole_tensor, numels)
        whole_tensor = None # release

        loss.backward()
        whole_grad = get_flat_grad(model)
        flat_grad = torch.empty_like(flat_param).requires_grad_(False).detach()

        # print0(f"flat_grad shape {flat_grad.shape}, whole_grad shape {whole_grad.shape}")
        dist.reduce_scatter_tensor(flat_grad, whole_grad, op=dist.ReduceOp.AVG)

        flat_param.grad = flat_grad
        optim.step()
        flat_param.grad = None
        flat_param.detach_()

        for param in model.parameters():
            param.grad = None

        # printall(f"hash after an iteration {compute_tensor_hash(flat_param)}")
    return flat_param
