# Example outputs:
#   (1024, 1048576), 2.674208 ms, 1.606071 tbgs
#   (1024, 1048573), 3.332944 ms, 1.288637 tbgs
#   (1024, 1048579), 3.323552 ms, 1.292286 tbgs

import torch
from torch._inductor.utils import do_bench
from torch._inductor import config

# config.benchmark_kernel = True

torch.set_default_device("cuda")

@torch.compile
def f(x):
    return x.sum(dim=-1)

for M, N in (
    (1024, 1024 * 1024),
    (1024, 1024 * 1024 - 3), # 1.25x slow down
    (1024, 1024 * 1024 + 3),
):
    x = torch.rand(M, N)
    ms = do_bench(lambda: f(x))
    total_bytes = M * N * 4
    print(f"({M}, {N}), {ms:3f} ms, {total_bytes / ms / 1e9:3f} tbgs")
print("bye")
