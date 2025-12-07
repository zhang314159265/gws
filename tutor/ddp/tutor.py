"""
A standalone tutorial for DDP.
"""
from torch import nn
import torch
from torch.nn.parallel.distributed import DistributedDataParallel as DDP
import torch.multiprocessing as mp
import torch.distributed as dist
import os

class MyModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.param = nn.Parameter(torch.randn(16))

    def forward(self, inp):
        return self.param * inp

def process_entry(rank):
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    m = MyModule().to(rank)
    m = DDP(m)
    inp = torch.ones(16).float().to(rank) * (rank + 1)
    m(inp).sum().backward()

    # Both rank should print [1.5, ..., 1.5] as gradient
    print(m.module.param.grad)
    dist.destroy_process_group()

world_size = 2

if __name__ == "__main__":
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12345"
    mp.spawn(process_entry, args=(), nprocs=world_size, join=True)
