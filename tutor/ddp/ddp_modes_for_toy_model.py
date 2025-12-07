"""
Apply various DDP modes to a toy model. Numericals are verified by
comparing the gradients hash on each rank.
"""

import os
from torch.nn.parallel.distributed import DistributedDataParallel as DDP
import torch.multiprocessing as mp
from torch import nn
import torch.nn.functional as F
import torch
from torch.optim import SGD
import torch.distributed as dist
import hashlib

def printall(msg):
    rank = torch.distributed.get_rank()
    print(f"Rank {rank}: {msg}")

class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(1024, 32, bias=False)
        self.linear2 = nn.Linear(32, 1, bias=False)

    def forward(self, x):
        x = self.linear1(x)
        x = F.relu(x)
        x = self.linear2(x)
        return F.sigmoid(x)
    
    def param_hash(self):
        """
        Compute the hash of all the parameters in this module
        """
        hasher = hashlib.sha256()
        for param in self.parameters():
            hasher.update(param.detach().cpu().numpy().tobytes())
        return hasher.hexdigest()[:8]

class DataGenerator:
    def generate(self, bs):
        return torch.randn([bs, 1024], device="cuda"), torch.randint(0, 2, [bs], device="cuda").float()

def process_main(rank):
    world_size = int(os.getenv("WORLD_SIZE"))
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

    torch.manual_seed(1000 + rank)
    torch.cuda.set_device(rank)
    print(f"current_device is {torch.cuda.current_device()}")

    model = MLP().to("cuda")
    model = DDP(model)
    datagen = DataGenerator()
    optim = SGD(model.parameters(), lr=0.01)

    for _ in range(2):
        # train for two steps
        x, label = datagen.generate(32)
        probs = model(x)
        loss = F.binary_cross_entropy(probs.flatten(), label)
        loss.backward()
        optim.step()
        optim.zero_grad(set_to_none=True)

    printall(model.module.param_hash())
    dist.destroy_process_group()

if __name__ == "__main__":
    world_size = 2
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12345"
    os.environ["WORLD_SIZE"] = str(world_size)
    mp.spawn(process_main, args=(), nprocs=world_size, join=True)
