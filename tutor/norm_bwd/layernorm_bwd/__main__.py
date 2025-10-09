import torch
import torch._inductor.config as inductor_config
from functools import partial
import torch.nn.functional as F

from .ref import ref_fwd, ref_bwd
from .triton import triton_bwd
from .triton_fused_loop import triton_fused_loop_bwd
from .triton_fused_1pass import triton_fused_1pass_bwd
from ..bench import assert_close, bench

def check_and_bench(fn):
    grads = fn(x, w, b, mean, rstd, dy, None)
    assert_close(ref_grads, grads)
    bench(fn.__name__, lambda: fn(x, w, b, mean, rstd, dy, None))

# setup inputs
torch.manual_seed(1337)
eps = 1e-5
M = 1152 * 500
N = 384
dtype = torch.bfloat16
wbdtype = torch.bfloat16
x = torch.randn(M, N, dtype=dtype, device="cuda", requires_grad=True)
w = torch.randn(N, dtype=wbdtype, device="cuda", requires_grad=True)
b = torch.randn(N, dtype=wbdtype, device="cuda", requires_grad=True)
dy = torch.rand_like(x)

# run reference
ref_y, mean, rstd = ref_fwd(x, w, b, eps, check=False)
ref_grads = ref_bwd(x, w, b, mean, rstd, dy, ref_y)

total_bytes = x.nbytes * 2 + w.nbytes * 2 + b.nbytes + mean.nbytes + rstd.nbytes + dy.nbytes

# run inductor
opt_ref_fwd = torch.compile(ref_fwd)
inductor_y = opt_ref_fwd(x, w, b, eps, return_mean_rstd=False)
inductor_grads = ref_bwd(x, w, b, mean, rstd, dy, inductor_y)
assert_close(ref_grads, inductor_grads)

bench = partial(bench, total_bytes=total_bytes)
bench("baseline", lambda: ref_bwd(x, w, b, mean, rstd, dy, ref_y))
bench("inductor", lambda: ref_bwd(x, w, b, mean, rstd, dy, inductor_y))
check_and_bench(triton_fused_1pass_bwd)
check_and_bench(triton_fused_loop_bwd)
check_and_bench(triton_bwd)

print("bye")
