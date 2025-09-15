from cutlass import cute
from cutlass.cute.runtime import from_dlpack
import torch
import cutlass
from functools import partial

@cute.kernel
def naive_elementwise_add_kernel(
    gA: cute.Tensor,
    gB: cute.Tensor,
    gC: cute.Tensor,
):
    tidx, _, _ = cute.arch.thread_idx()
    bidx, _, _ = cute.arch.block_idx()
    bdim, _, _ = cute.arch.block_dim()

    thread_idx = bidx * bdim + tidx

    m, n = gA.shape
    mi = thread_idx // n
    ni = thread_idx % n

    a_val = gA[mi, ni]
    b_val = gB[mi, ni]

    gC[mi, ni] = a_val + b_val

@cute.jit
def naive_elementwise_add(
    mA: cute.Tensor,
    mB: cute.Tensor,
    mC: cute.Tensor,
):
    num_threads_per_block = 256

    m, n = mA.shape
    kernel = naive_elementwise_add_kernel(mA, mB, mC)
    kernel.launch(grid=((m * n) // num_threads_per_block, 1, 1),
        block=(num_threads_per_block, 1, 1))

# Data get cached
factor = 8
M, N = 2048 * factor, 2048 * factor

a = torch.randn(M, N, device="cuda", dtype=torch.float16)
b = torch.randn(M, N, device="cuda", dtype=torch.float16)
c = torch.zeros(M, N, device="cuda", dtype=torch.float16)

a_ = from_dlpack(a, assumed_align=16)
b_ = from_dlpack(b, assumed_align=16)
c_ = from_dlpack(c, assumed_align=16)

naive_elementwise_add_ = cute.compile(naive_elementwise_add, a_, b_, c_)
naive_elementwise_add_(a_, b_, c_)

torch.testing.assert_close(c, a + b)

def benchmark(callable, *, num_warmups, num_iterations):
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    torch.cuda.synchronize()

    for _ in range(num_warmups):
        callable()

    start_event.record(stream=torch.cuda.current_stream())
    for _ in range(num_iterations):
        callable()
    end_event.record(stream=torch.cuda.current_stream())
    torch.cuda.synchronize()

    elapsed_time = start_event.elapsed_time(end_event)
    avg_time = elapsed_time / num_iterations

    print(f"Average execution time: {avg_time:.4f} ms")
    print(f"Throughput {(3 * a.numel() * 2) / (avg_time / 1000) / 1e9:.2f} GB/s")

benchmark(partial(naive_elementwise_add_, a_, b_, c_), num_warmups=5, num_iterations=100)

@cute.kernel
def vectorized_elementwise_add_kernel(
    gA: cute.Tensor,
    gB: cute.Tensor,
    gC: cute.Tensor,
):
    tidx, _, _ = cute.arch.thread_idx()
    bidx, _, _ = cute.arch.block_idx()
    bdim, _, _ = cute.arch.block_dim()

    thread_idx = bidx * bdim + tidx

    m, n = gA.shape[1]
    mi = thread_idx // n
    ni = thread_idx % n

    a_val = gA[(None, (mi, ni))].load()
    b_val = gB[(None, (mi, ni))].load()
    print(f"[DSL INFO] sliced gA = {gA[(None, (mi, ni))]}")
    print(f"[DSL INFO] sliced gB = {gB[(None, (mi, ni))]}")

    gC[(None, (mi, ni))] = a_val + b_val

@cute.jit
def vectorized_elementwise_add(
    mA: cute.Tensor,
    mB: cute.Tensor,
    mC: cute.Tensor,
):
    threads_per_block = 256

    tilesize = 4
    gA = cute.zipped_divide(mA, (1, tilesize))
    gB = cute.zipped_divide(mB, (1, tilesize))
    gC = cute.zipped_divide(mC, (1, tilesize))

    print(f"[DSL INFO] Tiled Tensors:")
    print(f"[DSL INFO]  gA  = {gA}")
    print(f"[DSL INFO]  gB  = {gB}")
    print(f"[DSL INFO]  gC  = {gC}")

    vectorized_elementwise_add_kernel(gA, gB, gC).launch(
        grid=(cute.size(gC, mode=[1]) // threads_per_block, 1, 1),
        block=(threads_per_block, 1, 1),
    )

a = torch.randn(M, N, device="cuda", dtype=torch.float16)
b = torch.randn(M, N, device="cuda", dtype=torch.float16)
c = torch.zeros(M, N, device="cuda", dtype=torch.float16)

a_ = from_dlpack(a, assumed_align=16)
b_ = from_dlpack(b, assumed_align=16)
c_ = from_dlpack(c, assumed_align=16)

compiled_func = cute.compile(vectorized_elementwise_add, a_, b_, c_)
compiled_func(a_, b_, c_)
torch.testing.assert_close(c, a + b)

benchmark(partial(compiled_func, a_, b_, c_), num_warmups=5, num_iterations=100)

@cute.kernel
def elementwise_add_kernel(
    gA: cute.Tensor,
    gB: cute.Tensor,
    gC: cute.Tensor,
    tv_layout: cute.Layout,
):
    tidx, _, _ = cute.arch.thread_idx()
    bidx, _, _ = cute.arch.block_idx()

    blk_coord = ((None, None), bidx)

    blkA = gA[blk_coord]
    blkB = gB[blk_coord]
    blkC = gC[blk_coord]

    tidfrgA = cute.composition(blkA, tv_layout)
    tidfrgB = cute.composition(blkB, tv_layout)
    tidfrgC = cute.composition(blkC, tv_layout)

    print(f"Composed with TV layout:")
    print(f"  tidfrgA: {tidfrgA.type}")

    # None represent slice of the entire per-thread data
    thr_coord = (tidx, None)

    thrA = tidfrgA[thr_coord]
    thrB = tidfrgB[thr_coord]
    thrC = tidfrgC[thr_coord]

    thrC[None] = thrA.load() + thrB.load()

@cute.jit
def elementwise_add(
    mA: cute.Tensor,
    mB: cute.Tensor,
    mC: cute.Tensor,
):
    thr_layout = cute.make_layout((4, 32), stride=(32, 1))
    val_layout = cute.make_layout((4, 8), stride=(8, 1))
    tiler_mn, tv_layout = cute.make_layout_tv(thr_layout, val_layout)
    print(f"Tiler: {tiler_mn}")
    print(f"TV Layout: {tv_layout}")

    gA = cute.zipped_divide(mA, tiler_mn)
    gB = cute.zipped_divide(mB, tiler_mn)
    gC = cute.zipped_divide(mC, tiler_mn)

    print("Tiled Input Tensors:")
    print(f"  gA: {gA.type}")
    print(f"  gB: {gB.type}")
    print(f"  gC: {gC.type}")

    elementwise_add_kernel(
        gA, gB, gC, tv_layout
    ).launch(
        grid=[cute.size(gC, mode=[1]), 1, 1],
        block=[cute.size(tv_layout, mode=[0]), 1, 1],
    )

a = torch.randn(M, N, device="cuda", dtype=torch.float16)
b = torch.randn(M, N, device="cuda", dtype=torch.float16)
c = torch.zeros(M, N, device="cuda", dtype=torch.float16)

a_ = from_dlpack(a, assumed_align=16)
b_ = from_dlpack(b, assumed_align=16)
c_ = from_dlpack(c, assumed_align=16)

elementwise_add_ = cute.compile(elementwise_add, a_, b_, c_)
elementwise_add_(a_, b_, c_)

torch.testing.assert_close(c, a + b)

benchmark(partial(elementwise_add_, a_, b_, c_), num_warmups=5, num_iterations=200)

@cute.kernel
def elementwise_apply_kernel(
    op: cutlass.Constexpr,
    gA: cute.Tensor,
    gB: cute.Tensor,
    gC: cute.Tensor,
    tv_layout: cute.Layout
):
    tidx, _, _ = cute.arch.thread_idx()
    bidx, _, _ = cute.arch.block_idx()

    blk_coord = ((None, None), bidx)

    blkA = gA[blk_coord]
    blkB = gB[blk_coord]
    blkC = gC[blk_coord]

    tidfrgA = cute.composition(blkA, tv_layout)
    tidfrgB = cute.composition(blkB, tv_layout)
    tidfrgC = cute.composition(blkC, tv_layout)

    print(f"Composed with TV layout:")
    print(f"  tidfrgA: {tidfrgA.type}")

    # None represent slice of the entire per-thread data
    thr_coord = (tidx, None)

    thrA = tidfrgA[thr_coord]
    thrB = tidfrgB[thr_coord]
    thrC = tidfrgC[thr_coord]

    thrC[None] = op(thrA.load(), thrB.load())

@cute.jit
def elementwise_op(
    op: cutlass.Constexpr,
    mA: cute.Tensor,
    mB: cute.Tensor,
    mC: cute.Tensor,
):
    thr_layout = cute.make_layout((4, 32), stride=(32, 1))
    val_layout = cute.make_layout((4, 8), stride=(8, 1))
    tiler_mn, tv_layout = cute.make_layout_tv(thr_layout, val_layout)
    print(f"Tiler: {tiler_mn}")
    print(f"TV Layout: {tv_layout}")

    gA = cute.zipped_divide(mA, tiler_mn)
    gB = cute.zipped_divide(mB, tiler_mn)
    gC = cute.zipped_divide(mC, tiler_mn)

    print("Tiled Input Tensors:")
    print(f"  gA: {gA.type}")
    print(f"  gB: {gB.type}")
    print(f"  gC: {gC.type}")

    elementwise_apply_kernel(
        op, gA, gB, gC, tv_layout
    ).launch(
        grid=[cute.size(gC, mode=[1]), 1, 1],
        block=[cute.size(tv_layout, mode=[0]), 1, 1],
    )

a = torch.randn(M, N, device="cuda", dtype=torch.float16)
b = torch.randn(M, N, device="cuda", dtype=torch.float16)
c = torch.zeros(M, N, device="cuda", dtype=torch.float16)

a_ = from_dlpack(a, assumed_align=16)
b_ = from_dlpack(b, assumed_align=16)
c_ = from_dlpack(c, assumed_align=16)

from operator import mul

elementwise_op(mul, a_, b_, c_)

torch.testing.assert_close(c, mul(a, b))
elementwise_op_ = cute.compile(elementwise_op, mul, a_, b_, c_)
benchmark(partial(elementwise_op_, a_, b_, c_), num_warmups=5, num_iterations=200)

def mul_relu(a, b):
    tmp = a * b
    return cute.where(tmp > 0, tmp, cute.full_like(tmp, 0))

def mul_relu_ref(a, b):
    tmp = a * b
    return torch.relu(tmp)

elementwise_op(mul_relu, a_, b_, c_)
torch.testing.assert_close(c, mul_relu_ref(a, b))
elementwise_op_mul_relu_ = cute.compile(elementwise_op, mul_relu, a_, b_, c_)
benchmark(partial(elementwise_op_mul_relu_, a_, b_, c_), num_warmups=5, num_iterations=200)

exit()

