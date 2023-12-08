# Benchmark outer reduction

import torch
from torch._inductor.utils import do_bench
from torch._inductor import config
from torch._dynamo.testing import rand_strided

config.benchmark_kernel = True

torch.set_default_device("cuda")

@torch.compile
def f(x):
    return x.sum(dim=-1)

for M, N in (
    (1024, 1024 * 1024),
    (1024, 1024 * 1024 - 3), # around 1x
    (1024, 1024 * 1024 + 3), # around 1x
    (1024 - 3, 1024 * 1024), # around 1.16x slower
    (1024 + 3, 1024 * 1024), # around 1.16x slower
):
    x = rand_strided((M, N), (1, M), device="cuda")
    ms = do_bench(lambda: f(x))
    total_bytes = M * N * 4
    print(f"({M}, {N}), {ms:3f} ms, {total_bytes / ms / 1e9:3f} tbgs")

print("bye")
