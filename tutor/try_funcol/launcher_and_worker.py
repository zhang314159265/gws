import torch.distributed._functional_collectives as _functional_collectives
import torch
import os
import subprocess
import sys
import torch.distributed as dist

rank = int(os.getenv("RANK")) if "RANK" in os.environ else None
world_size = int(os.getenv("WORLD_SIZE")) if "WORLD_SIZE" in os.environ else None

def launch():
    for i in range(2):
        env = dict(
            **os.environ,
            RANK=str(i),
            WORLD_SIZE=str(2),
            MASTER_ADDR="localhost",
            MASTER_PORT=str(12345),
        )
        env.pop("CUDA_VISIBLE_DEVICES")
        args = [
            sys.executable,
            __file__
        ]
        subprocess.Popen(args, env=env)

def work():
    print(rank, world_size)

    def func(a, *, tag="", ranks=(0, 1)):
        a = a + 1
        ar = _functional_collectives.all_reduce(a, "sum", list(ranks), tag)
        return ar * 2

    device = f"cuda:{rank}"
    dist.init_process_group("nccl")
    x = torch.ones(4, 4, device=device) + rank
    compiled = torch.compile(func)
    print(compiled(x))
    dist.destroy_process_group()
    

if rank is None:
    launch()
else:
    work()
