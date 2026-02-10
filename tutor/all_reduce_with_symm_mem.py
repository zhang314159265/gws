import torch
import torch.distributed._symmetric_memory as symm_mem
import torch.distributed as dist
import triton
import triton.language as tl

from kraken._ptx_utils import symm_mem_sync  # TODO

@triton.jit
def one_shot_all_reduce_kernel(
    buf_tuple,
    signal_pad_ptrs,
    output_ptr,
    numel: tl.constexpr,
    rank: tl.constexpr,
    world_size: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    # XXX the sync call are copied from Kraken. But commenting them
    # out also works.
    # symm_mem_sync(signal_pad_ptrs, None, rank, world_size, hasSubsequentMemAccess=True)
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE

    while block_start < numel:
        offsets = block_start + tl.arange(0, BLOCK_SIZE)
        acc = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)

        for i in tl.static_range(world_size):
            buffer_rank = buf_tuple[i]
            x = tl.load(buffer_rank + offsets)
            acc += x
        tl.store(output_ptr + offsets, acc)
        block_start += tl.num_programs(axis=0) * BLOCK_SIZE

    # symm_mem_sync(signal_pad_ptrs, None, rank, world_size, hasPreviousMemAccess=True)

def one_shot_all_reduce(x):
    symm_mem_hdl = symm_mem.rendezvous(x, dist.group.WORLD)
    buf_tuple = tuple(
        symm_mem_hdl.get_buffer(i, tuple(x.shape), x.dtype)
        for i in range(symm_mem_hdl.world_size)
    )
    BLOCK_SIZE = 1024
    assert x.numel() % BLOCK_SIZE == 0
    output = torch.empty_like(x)
    one_shot_all_reduce_kernel[(x.numel() // BLOCK_SIZE, 1, 1)](
        buf_tuple,
        symm_mem_hdl.signal_pad_ptrs_dev,
        output,
        output.numel(),
        rank=symm_mem_hdl.rank,
        world_size=symm_mem_hdl.world_size,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    return output

def main(tol=1e-4):
    dist.init_process_group()
    rank = dist.get_rank()
    print(f"rank = {rank}")
    t = symm_mem.empty(
        4096,
        device=f"cuda:{rank}",
    )
    t.normal_()

    act = one_shot_all_reduce(t)

    ref = t.clone()
    dist.all_reduce(ref)

    torch.testing.assert_close(ref, act, atol=tol, rtol=tol)
    dist.destroy_process_group()

if __name__ == "__main__":
    main()
    print("PASS")
