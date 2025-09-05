# The recommended way to start workers with torchrun:
#   CUDA_VISIBLE_DEVICES=2,3 torchrun --nproc-per-node 2 tutor/try_funcol/worker_only.py
import os
import torch.distributed as dist
import torch
from torch.distributed import _functional_collectives

rank = int(os.getenv("RANK"))
world_size = int(os.getenv("WORLD_SIZE"))
ranks = list(range(world_size))
tag = ""

def work():
    print(rank, world_size)

    def func(a):
        a = a + 1
        ar = _functional_collectives.all_reduce(a, "sum", ranks, tag)
        return ar * 2

    device = f"cuda:{rank}"
    torch.cuda.set_device(device)
    dist.init_process_group("nccl")
    x = torch.ones(4, 4, device=device) + rank
    compiled = torch.compile(func)
    print(compiled(x))
    dist.destroy_process_group()

work()
