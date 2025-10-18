import torch
import torch.nn.functional as F
from .ref import ref_attn_fwd, ref_attn_bwd
from .torch_trivial import attn_fwd as torch_trivial_attn_fwd, attn_bwd as torch_trivial_attn_bwd
from triton.testing import do_bench

def assert_close(ref, act, atol=1e-2, rtol=1e-2):
    torch.testing.assert_close(ref, act, atol=atol, rtol=rtol)

torch.manual_seed(1337)

# use torch.bfloat16 results in failure in assert_close.
Q, K, V = [torch.randn(32, 1024, 12, 64, dtype=torch.float16, device="cuda", requires_grad=True) for _ in range(3)]
dY = torch.randn_like(Q)

ref_fwd_out = ref_attn_fwd(Q, K, V)
Y = ref_fwd_out
ref_bwd_out = ref_attn_bwd(dY, Q, K, V, Y)

eager_fwd_ms = do_bench(lambda: ref_attn_fwd(Q, K, V))
eager_bwd_ms = do_bench(lambda: ref_attn_bwd(dY, Q, K, V, Y))
print(f"{eager_fwd_ms=}")
print(f"{eager_bwd_ms=}")

if True:  # torch_trivial
    # TODO: figure out why torch_trivial is faster than eager
    torch_trivial_fwd_out = torch_trivial_attn_fwd(Q, K, V)
    assert_close(ref_fwd_out, torch_trivial_fwd_out)
    torch_trivial_fwd_ms = do_bench(lambda: torch_trivial_attn_fwd(Q, K, V))
    print(f"{torch_trivial_fwd_ms=}")
    
    torch_trivial_bwd_out = torch_trivial_attn_bwd(dY, Q, K, V, Y)
    assert_close(ref_bwd_out, torch_trivial_bwd_out)
    torch_trivial_bwd_ms = do_bench(lambda: torch_trivial_attn_bwd(dY, Q, K, V, Y))
    print(f"{torch_trivial_bwd_ms=}")

print("bye")
