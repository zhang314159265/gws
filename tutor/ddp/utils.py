import torch

def printall(msg):
    rank = torch.distributed.get_rank()
    print(f"Rank {rank}: {msg}")

def print0(msg):
    rank = torch.distributed.get_rank()
    if rank == 0:
        print(f"Rank {rank}: {msg}")


