import torch
from util import checkclose, bench
import curun
from memcpy_triton import memcpy_triton
from memcpy_cudart import memcpy_cudart

x = torch.randn(2 ** 30, device="cuda")
act = torch.empty_like(x)

def baseline(src, dst):
    dst.copy_(src)

@torch.compile
def compiled_memcpy(src, dst):
    dst.copy_(src)

cukernel = curun.open("out/memcpy.cubin").sym("memcpy_kernel")
def memcpy_with_cuda(src, dst):
    cukernel[1024, 1024](src, dst, src.numel())

act.zero_()
baseline(x, act)
assert x is not act
checkclose(x, act)

act.zero_()
compiled_memcpy(x, act)
checkclose(x, act)

act.zero_()
memcpy_triton(x, act)
checkclose(x, act)

act.zero_()
memcpy_cudart(x, act)
torch.cuda.synchronize()
checkclose(x, act)

act.zero_()
memcpy_with_cuda(x, act)
checkclose(x, act)

total_bytes = x.numel() * x.itemsize * 2
bench(lambda: baseline(x, act), "baseline", total_bytes=total_bytes)
bench(lambda: compiled_memcpy(x, act), "compiled", total_bytes=total_bytes)
bench(lambda: memcpy_triton(x, act), "pure_triton", total_bytes=total_bytes)
bench(lambda: memcpy_cudart(x, act), "cudart", total_bytes=total_bytes)
bench(lambda: memcpy_with_cuda(x, act), "cuda_kernel", total_bytes=total_bytes)

print("memcpy bye")
