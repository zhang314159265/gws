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
from .util import compute_tensor_hash

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
    0: "2826b6e1",
    1: "ee4a6cb3",
}

def manually_compute_shard_hash(model, world_size):
    params = list(model.parameters())
    params = [p.flatten() for p in params]
    full = torch.cat(params)
    assert full.numel() % world_size == 0
    shards = full.chunk(world_size)
    for i, shard in enumerate(shards):
        printall(f"shard {i} hash {compute_tensor_hash(shard)}")

def process_main(rank):
    world_size = int(os.getenv("WORLD_SIZE"))
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

    torch.manual_seed(1000 + rank)
    torch.cuda.set_device(rank)
    print(f"current_device is {torch.cuda.current_device()}")

    model = MLP().to("cuda")
    # printall(f"model hash before applying fsdp {model.param_hash()}")
    # manually_compute_shard_hash(model, world_size)
    mode = os.getenv("mode", "raw")
    print0(f"mode is {mode}")
    datagen = DataGenerator()
    lr = 0.01
    if mode == "torch":
        printall(f"param hash before wrapping with FSDP: {model.param_hash()}")
        model = FSDP(model)
        optim = SGD(model.parameters(), lr=lr)
        printall(f"param hash before training {model.module.param_hash()}")
        for _ in range(2):
            train_step(model, datagen, optim)
            printall(f"param hash after an iteration {model.module.param_hash()}")

        assert model.module.param_hash() == expected_hash[rank]
    elif mode == "raw":
        from .raw import raw_fsdp
        flat_param = raw_fsdp(model, datagen, lr=lr)

        assert compute_tensor_hash(flat_param) == expected_hash[rank]
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
