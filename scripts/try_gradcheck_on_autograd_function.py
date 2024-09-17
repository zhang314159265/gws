import torch

class MySin(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return x.sin()

    @staticmethod
    def backward(ctx, grad_out):
        x, = ctx.saved_tensors
        return grad_out * x.cos()

x = torch.randn(1024, requires_grad=True, device="cuda")
MySin.apply(x).sum().backward()

# gradcheck needs double tensor as input
torch.autograd.gradcheck(MySin.apply, x.double())
print("bye")
