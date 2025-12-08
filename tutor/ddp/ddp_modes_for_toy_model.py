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
from .utils import printall, print0
from .train_step import train_step
from .my_ddp import MyDDP

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

    mode = os.getenv("mode", "myddp")
    expected_hash = "a43a3bd5"  # need change if seed/#gpu/model/inputs etc changed
    print0(f"mode is {mode}")
    datagen = DataGenerator()
    optim = SGD(model.parameters(), lr=0.01)
    if mode == "torch":
        model = DDP(model)
    
        for _ in range(2):
            # train for two steps
            train_step(model, datagen, optim)
   
        assert model.module.param_hash() == expected_hash
    elif mode == "raw":
        from .raw import raw_ddp
        raw_ddp(model, datagen, optim)
        assert model.param_hash() == expected_hash, f"Actual hash: {model.param_hash()}"
    elif mode == "myddp":
        model = MyDDP(model)

        for _ in range(2):
            train_step(model, datagen, optim)

        assert model.module.param_hash() == expected_hash
    else:
        raise RuntimeError(f"Unrecognized mode: {mode}")

    printall("PASS!")
    dist.destroy_process_group()

if __name__ == "__main__":
    world_size = 2
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12345"
    os.environ["WORLD_SIZE"] = str(world_size)
    mp.spawn(process_main, args=(), nprocs=world_size, join=True)
