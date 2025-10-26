import torch
import cutlass
from cutlass import cute
from cutlass.cute.runtime import from_dlpack

@cute.kernel
def kernel(gX: cute.Tensor, gY: cute.Tensor, tv_layout: cute.Layout):
    bidx, _, _ = cute.arch.block_idx()
    tidx, _, _ = cute.arch.thread_idx()

    gX = gX[None, bidx]
    gY = gY[None, bidx]
    gX = cute.composition(gX, tv_layout)
    gY = cute.composition(gY, tv_layout)

    gY[tidx, None] = (gX[tidx, None].load() + 1.0).to(cute.BFloat16)

@cute.jit
def plus1(mX: cute.Tensor, mY: cute.Tensor):
    thr_layout = cute.make_ordered_layout((4, 32), order=(1, 0))
    val_layout = cute.make_ordered_layout((2, 16), order=(1, 0))
    val_layout = cute.recast_layout(mX.element_type.width, 8, val_layout)
    tiler_mn, tv_layout = cute.make_layout_tv(thr_layout, val_layout)
    gX = cute.zipped_divide(mX, tiler_mn)
    gY = cute.zipped_divide(mY, tiler_mn)
    kernel(gX, gY, tv_layout).launch(
        grid=(cute.size(gX, mode=[1]), 1, 1),
        block=(cute.size(tv_layout, mode=[0]), 1, 1),
    )


x = torch.randn(1024 * 10, 2048, device="cuda", dtype=torch.bfloat16)
y = torch.empty_like(x)
x_ = from_dlpack(x, assumed_align=16)
y_ = from_dlpack(y, assumed_align=16)
compiled_plus1 = cute.compile(plus1, x_, y_)
compiled_plus1(x_, y_)

torch.testing.assert_close(x + 1, y)

us = cute.testing.benchmark(
    compiled_plus1,
    kernel_arguments=cute.testing.JitArguments(x_, y_),
)

tot_bytes = x.numel() * x.itemsize * 2
print(f"Latency {us * 1e-3:.3f} ms, throughput {tot_bytes / (us * 1e3) :.3f} GB/s")

print("bye")
