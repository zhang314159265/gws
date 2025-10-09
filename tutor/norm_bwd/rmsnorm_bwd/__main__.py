import torch
import torch._inductor.config as inductor_config

from ..bench import assert_close, bench
from .ref import ref_fwd, ref_bwd
from .triton import triton_bwd
from .triton_fused_loop import triton_fused_loop_bwd
from .triton_fused_1pass import triton_fused_1pass_bwd
from functools import partial

def check_and_bench(fn):
    grads = fn(x, w, rsqrt, dy, None)
    assert_close(ref_grads, grads)
    bench(fn.__name__, lambda: fn(x, w, rsqrt, dy, None))

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


# run inductor
opt_ref_fwd = torch.compile(ref_fwd)
inductor_y = opt_ref_fwd(x, w, eps, return_rsqrt=False)
inductor_grads = ref_bwd(x, w, rsqrt, dy, inductor_y)
assert_close(ref_grads, inductor_grads)

bench = partial(bench, total_bytes=total_bytes)
bench("baseline", lambda: ref_bwd(x, w, rsqrt, dy, ref_y))
bench("inductor", lambda: ref_bwd(x, w, rsqrt, dy, inductor_y))
check_and_bench(triton_fused_1pass_bwd)
check_and_bench(triton_fused_loop_bwd)
check_and_bench(triton_bwd)

print("bye")
