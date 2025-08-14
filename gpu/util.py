import torch

libkernel = torch.library.Library("kernel", "FRAGMENT")

def register_custom_op(name):
    def f_meta(x):
        return torch.empty_like(x)

    def decorator(f):
        libkernel.define(f"{name}(Tensor self) -> Tensor", tags=())
        libkernel.impl(name, f, "CUDA")
        libkernel.impl(name, f_meta, "Meta")
        
        return f
    return decorator
