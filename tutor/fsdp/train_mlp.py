from torch import nn
import os
import torch.multiprocessing as mp
import torch.distributed as dist
import torch
from torch.optim import SGD
import torch.nn.functional as F
import hashlib
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.nn.parallel.distributed import DistributedDataParallel as DDP
from ..ddp.utils import print0, printall
from ..ddp.train_step import train_step

class DataGenerator:
    def generate(self, bs):
        return torch.randn([bs, 1024], device="cuda"), torch.randint(0, 2, [bs], device="cuda").float()

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

expected_hash = {
    0: "6a20b23f",
    1: "cfc9a384",
}

def process_main(rank):
    world_size = int(os.getenv("WORLD_SIZE"))
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

    torch.manual_seed(1000 + rank)
    torch.cuda.set_device(rank)
    print(f"current_device is {torch.cuda.current_device()}")

    model = MLP().to("cuda")
    mode = os.getenv("mode", "torch")
    print0(f"mode is {mode}")
    datagen = DataGenerator()
    optim = SGD(model.parameters(), lr=0.01)
    if mode == "torch":
        model = FSDP(model)
        for _ in range(2):
            train_step(model, datagen, optim)

        printall(f"param hash after 2 training steps {model.module.param_hash()}")
        assert model.module.param_hash() == expected_hash[rank]
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
