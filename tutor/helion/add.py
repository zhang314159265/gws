"""
-- benchmark result for (1024, 1024) --
autotune_effort="none": 0.012 ms, 1.029 tbgs
autotune_effort="full": 0.013 ms, 0.995 tbgs

-- SIZE_SCALE=64
effort="none": 11.740 ms, 4.390 tbgs
effort="full": OOM

-- SIZE_SCALE=32
effort="none": 2.381 ms, 5.412 tbgs
    @helion.kernel(config=helion.Config(block_sizes=[32, 32], flatten_loops=[False], indexing=['pointer', 'pointer', 'pointer'], l2_groupings=[1], load_eviction_policies=['', ''], loop_orders=[[0, 1]], num_stages=1, num_warps=4, pid_type='flat', range_flattens=[None], range_multi_buffers=[None], range_num_stages=[0], range_unroll_factors=[0], range_warp_specializes=[None]), static_shapes=True)

block-size [2, 2048]: 1.924 ms, 6.698 tbgs

use tensor_descriptors for 2 of the memory access: generated triton code does not change (because tensor descriptor is skipped due to the 'weird' block size steup, change block size fo [32, 256] enables tensor descriptor)

Change l2_groups to 2: 1.850 ms, 6.965 tbgs (1.04x up)
Change load eviction polity to ['', 'last']: 1.846 ms, 6.980 tbgs

Change num_stages to 6: 1.842 ms, 6.993 tbgs
  not important. Change it to 1 upon the autotuner picked best config: 1.821 ms, 7.074 tbgs
Change num_warps to 16: 1.825 ms, 7.062 tbgs

effort="full": 1.839 ms, 7.005 tbgs
    @helion.kernel(config=helion.Config(block_sizes=[2, 2048], flatten_loops=[False], indexing=['tensor_descriptor', 'pointer', 'tensor_descriptor'], l2_groupings=[2], load_eviction_policies=['', 'last'], loop_orders=[[0, 1]], num_stages=6, num_warps=16, pid_type='flat', range_flattens=[None], range_multi_buffers=[None], range_num_stages=[0], range_unroll_factors=[0], range_warp_specializes=[None]), static_shapes=True)
"""

import torch
import helion
import helion.language as hl
from triton.testing import do_bench
import os


@helion.kernel(config=helion.Config(block_sizes=[2, 2048], flatten_loops=[False], indexing=['tensor_descriptor', 'pointer', 'tensor_descriptor'], l2_groupings=[2], load_eviction_policies=['', 'last'], loop_orders=[[0, 1]], num_stages=6, num_warps=16, pid_type='flat', range_flattens=[None], range_multi_buffers=[None], range_num_stages=[0], range_unroll_factors=[0], range_warp_specializes=[None]), static_shapes=True)
def helion_add(x, y):
    # TODO broadcasting and type promotion
    z = torch.empty_like(x)
    for tile in hl.tile(z.shape):
        z[tile] = x[tile] + y[tile]

    return z

size_scale = int(os.getenv("SIZE_SCALE", "32"))
x, y = [torch.randn(1024 * size_scale, 1024 * size_scale, device="cuda") for _ in range(2)]
ref = x + y
act = helion_add(x, y)
torch.testing.assert_close(ref, act)

for _ in range(3): # warmup
    helion_add(x, y)

ms = do_bench(lambda: helion_add(x, y))
tot_bytes = x.nbytes * 3
tbgs = (tot_bytes / 1e12) / (ms / 1e3)
print(f"{ms:.3f} ms, {tbgs:.3f} tbgs")
print("PASS")
