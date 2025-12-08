from torch import nn
import torch.distributed as dist
from .utils import printall
import enum

class HookType(enum.Enum):
    TENSOR_HOOK = 0
    MODULE_HOOK = 1

    def __str__(self):
        return self.name

def _param_grad_hook(grad):
    # printall(f"param grad hook called for {id(grad)}")
    dist.all_reduce(grad, op=dist.ReduceOp.AVG)
    return grad

def _module_bwd_hook(module, grad_input, grad_output):
    dist.breakpoint()
    return None 

class MyDDP(nn.Module):
    def __init__(self, module, hook_type=HookType.TENSOR_HOOK):
        super().__init__()
        self.module = module

        print(f"hook_type is {hook_type}")

        for param in module.parameters():
            dist.broadcast(param.detach(), src=0)

            if hook_type == HookType.TENSOR_HOOK:
                param.register_hook(_param_grad_hook)

        # XXX MODULE_HOOK does not work yet. The hook does not get
        # fired.
        if hook_type == HookType.MODULE_HOOK:
            module.register_full_backward_hook(_module_bwd_hook)

    def forward(self, *args, **kwargs):
        return self.module.forward(*args, **kwargs)
