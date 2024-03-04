import ptxrunner
import torch
import os
torch.set_default_device("cuda")

def add_args_and_ref():
    x = torch.randn(1024)
    y = torch.randn(1024)
    ref = x + y
    return (x, y, torch.zeros_like(ref), x.numel(), ref), 2

def sum_args_and_ref():
    x = torch.randn(1024, 1024)
    ref = x.sum(dim=-1)
    return (x, torch.zeros_like(ref), 1024, 1024, ref), 1

def dot_args_and_ref():
    N = 32
    a = torch.rand(N, N)
    b = torch.rand(N, N)
    ref = a @ b + 5
    return (a, b, torch.zeros_like(ref), ref), 2

for ptx_file, f_args_and_ref, gridDim, blockDim, shared in (
    ["add.ptx", add_args_and_ref, 32, 32*4, 0],
    ["sum.ptx", sum_args_and_ref, 512, 32*4, 2064],
    ["dot.ptx", dot_args_and_ref, 1, 32*4, 8192],
):
    print(f"-> Start testing {ptx_file}")
    (*args, ref), act_idx = f_args_and_ref()
    act = args[act_idx]

    with open(os.path.join("example_ptx", ptx_file)) as f:
        ptx_code = f.read()

    ptxrunner.load_and_run(
        ptx_code, args=args, gridDim=gridDim, blockDim=blockDim, shared=shared
    )
    tol = {"atol": 1e-3, "rtol": 1e-3}
    assert torch.allclose(ref, act, **tol), f"ref:\n{ref}\nact:\n{act}"
    print(f"<- Done testing {ptx_file}")

print("bye")
