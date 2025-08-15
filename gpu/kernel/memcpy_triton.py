import triton
import triton.language as tl

@triton.jit
def _kernel(src, dst, N, BLOCK: tl.constexpr):
    idx = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    x = tl.load(src + idx)
    tl.store(dst + idx, x)

def memcpy_triton(src, dst):
    BLOCK = 512
    _kernel[(src.numel() // BLOCK,)](src, dst, src.numel(), BLOCK)
