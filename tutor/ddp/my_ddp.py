from torch import nn
import torch.distributed as dist
from .utils import printall

def _param_grad_hook(grad):
    # printall(f"param grad hook called for {id(grad)}")
    dist.all_reduce(grad, op=dist.ReduceOp.AVG)
    return grad

class MyDDP(nn.Module):
    def __init__(self, module):
        super().__init__()
        self.module = module

        for param in module.parameters():
            dist.broadcast(param.detach(), src=0)
            param.register_hook(_param_grad_hook)

    def forward(self, *args, **kwargs):
        return self.module.forward(*args, **kwargs)
