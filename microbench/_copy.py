import torch
from triton.testing import do_bench

N = 10 ** 9
x = torch.randn(N, device="cuda")  # 4GB
y = torch.empty(N, device="cuda")

@torch.compile
def f(x, y):
    y.copy_(x)

if __name__ == "__main__":
    ms = do_bench(lambda: f(x, y))
    nbytes = x.nbytes + y.nbytes
    membw = nbytes / (ms / 1000)
    print(f"{ms=} nbytes={nbytes / 1e9:.3f}GB membw={membw / 1e12:.3f}TBGS")
