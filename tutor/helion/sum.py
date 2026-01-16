"""

-- shape [1024 * 1024, 1024]

default config: 0.620 ms, 6.937 tbgs
    @helion.kernel(config=helion.Config(block_sizes=[1], indexing=['pointer', 'pointer', 'pointer'], load_eviction_policies=['', ''], num_stages=1, num_warps=4, pid_type='flat', range_flattens=[None], range_multi_buffers=[None], range_num_stages=[0], range_unroll_factors=[0], range_warp_specializes=[None], reduction_loops=[None]), static_shapes=True)

block_sizes->[16]: 0.631 ms, 6.811 tbgs

change eviction policy: 0.626 ms, 6.872 tbgs

change stage/num-warps: 0.653 ms, 6.587 tbgs

change reduction_loops to [64]: 0.611 ms, 7.032 tbgs

full autotune picks: 0.612 ms, 7.024 tbgs
    @helion.kernel(config=helion.Config(block_sizes=[16], indexing=['pointer', 'pointer', 'pointer'], load_eviction_policies=['first', ''], num_stages=5, num_warps=16, pid_type='flat', range_flattens=[None], range_multi_buffers=[None], range_num_stages=[0], range_unroll_factors=[0], range_warp_specializes=[None], reduction_loops=[64]), static_shapes=True)

-- shape [2, 1024 * 1024 * 128]

helion full autotune: crash due to IMA
Inductor: 0.172 ms, 6.245 tbgs

-- shape [2, 1024 * 1024 * 32]

helion full autotune: helion: 0.619 ms, 0.434 tbgs . Pick the following config by autotuner:
    @helion.kernel(config=helion.Config(block_sizes=[1], indexing=['tensor_descriptor', 'tensor_descriptor', 'pointer'], load_eviction_policies=['', ''], num_stages=6, num_warps=32, pid_type='persistent_interleaved', range_flattens=[False], range_multi_buffers=[None], range_num_stages=[4], range_unroll_factors=[3], range_warp_specializes=[None], reduction_loops=[4096]), static_shapes=True)

Inductor: 0.060 ms, 4.464 tbgs

    XXX helion does not do split reduction (rev 71a04b1b927, Dec 31 2025)
"""

import helion
import helion.language as hl
import torch
from triton.testing import do_bench

torch.manual_seed(1337)

@helion.kernel(config=helion.Config(block_sizes=[1], indexing=['tensor_descriptor', 'tensor_descriptor', 'pointer'], load_eviction_policies=['', ''], num_stages=6, num_warps=32, pid_type='persistent_interleaved', range_flattens=[False], range_multi_buffers=[None], range_num_stages=[4], range_unroll_factors=[3], range_warp_specializes=[None], reduction_loops=[4096]), static_shapes=True)
def helion_sum(x):
    m, n = x.shape
    y = torch.empty(m, device=x.device, dtype=x.dtype)

    for tile in hl.tile(m):
        y[tile] = x[tile, :].sum(dim=-1)
    return y

x = torch.randn(2, 1024 * 1024 * 32, device="cuda")
ref = torch.sum(x, dim=-1)
act = helion_sum(x)
torch.testing.assert_close(ref, act, atol=1e-4, rtol=1e-4)

for _ in range(3):  # warmup
    helion_sum(x)

ms = do_bench(lambda: helion_sum(x))
tot_bytes = x.nbytes + act.nbytes
tbgs = (tot_bytes / 1e12) / (ms / 1e3)
print(f"helion: {ms:.3f} ms, {tbgs:.3f} tbgs")

@torch.compile
def inductor_sum(x):
    return torch.sum(x, dim=-1)

for _ in range(3):  # warmup
    inductor_sum(x)

ms = do_bench(lambda: inductor_sum(x))
tbgs = (tot_bytes / 1e12) / (ms / 1e3)
print(f"Inductor: {ms:.3f} ms, {tbgs:.3f} tbgs")
print("PASS sum")
