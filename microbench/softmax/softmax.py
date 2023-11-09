import torch

torch.set_default_device("cuda")

@torch.compile
def f(x):
    return x.softmax(dim=-1)

f(torch.randn(8192, 65536))
