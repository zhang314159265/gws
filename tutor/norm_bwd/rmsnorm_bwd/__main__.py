import torch
import torch._inductor.config as inductor_config

from ..bench import assert_close, bench
from .ref import ref_fwd, ref_bwd
from .triton import triton_bwd
from .triton_fused_loop import triton_fused_loop_bwd
from .triton_fused_1pass import triton_fused_1pass_bwd
from .triton_fused_1pass_smallload import triton_fused_1pass_smallload_bwd
from functools import partial

def liger_bwd(x, w, rsqrt, dy, y):
    from liger_kernel.ops.rms_norm import rms_norm_backward as _liger_bwd
    from liger_kernel.ops.rms_norm import _CASTING_MODE_LLAMA, _CASTING_MODE_GEMMA
    from liger_kernel.ops.utils import calculate_settings
    # adaptor for liger kernel
    # casting_mode = _CASTING_MODE_LLAMA.value # numerical issue for dw
    casting_mode = _CASTING_MODE_GEMMA.value
    block_size, num_warps = calculate_settings(x.size(-1))
    row_mode = None
    in_place = True # inplace mode is slightly faster
    dx, dw = _liger_bwd(dy, x, w, rsqrt, offset=0.0, casting_mode=casting_mode, BLOCK_SIZE=block_size, num_warps=num_warps, in_place=in_place, row_mode=row_mode)
    return dx, dw

def check_and_bench(fn, name=None):
    grads = fn(x, w, rsqrt, dy, None)
    # breakpoint()
    assert_close(ref_grads, grads)
    bench(name or fn.__name__, lambda: fn(x, w, rsqrt, dy, None))

def check_and_bench_inductor(name, mode):
    torch._dynamo.reset()
    opt_ref_fwd = torch.compile(ref_fwd, mode=mode)
    inductor_y = opt_ref_fwd(x, w, eps, return_rsqrt=False)
    inductor_grads = ref_bwd(x, w, rsqrt, dy, inductor_y)
    assert_close(ref_grads, inductor_grads)
    
    bench(name, lambda: ref_bwd(x, w, rsqrt, dy, inductor_y))

# setup inputs
torch.manual_seed(1337)
eps = 1e-5
M = 1152 * 500
N = 384
dtype = torch.bfloat16
x = torch.randn(M, N, dtype=dtype, device="cuda", requires_grad=True)
w = torch.randn(N, dtype=torch.float, device="cuda", requires_grad=True)
dy = torch.rand_like(x)

# run reference
ref_y, rsqrt = ref_fwd(x, w, eps)
ref_grads = ref_bwd(x, w, rsqrt, dy, ref_y)

total_bytes = x.nbytes * 2 + w.nbytes * 2 + dy.nbytes + rsqrt.nbytes

bench = partial(bench, total_bytes=total_bytes)
bench("baseline", lambda: ref_bwd(x, w, rsqrt, dy, ref_y))
check_and_bench_inductor("inductor", "default")
check_and_bench_inductor("inductor_max_autotune", "max-autotune")

dy_clone = dy.clone()
check_and_bench(liger_bwd) # out of box perf is very bad. Need enlarge grid size by 32x
dy = dy_clone
check_and_bench(triton_fused_1pass_bwd)
check_and_bench(triton_fused_1pass_smallload_bwd)
check_and_bench(triton_fused_loop_bwd)
check_and_bench(triton_bwd)

print("bye")
