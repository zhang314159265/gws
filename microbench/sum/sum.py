# Benchmark inner reduction
# Example outputs:
#   (1024, 1048576), 2.674208 ms, 1.606071 tbgs
#   (1024, 1048573), 3.332944 ms, 1.288637 tbgs
#   (1024, 1048579), 3.323552 ms, 1.292286 tbgs

import torch
from torch._inductor.utils import do_bench
from torch._inductor import config
from torch._dynamo import config as dynamo_config

dynamo_config.automatic_dynamic_shapes = False
config.benchmark_kernel = True

torch.set_default_device("cuda")

# XXX _load_kernel has been removed
# from torch._inductor import codecache
# kernel_paths = []
# orig_load_kernel = codecache._load_kernel
# def _mock_load_kernel(*args):
#     kernel = orig_load_kernel(*args)
#     global kernel_paths
#     kernel_paths.append(kernel.fn.fn.__code__.co_filename)
#     return kernel
# 
# codecache._load_kernel = _mock_load_kernel

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


# print(f"kernel paths: {kernel_paths}")
