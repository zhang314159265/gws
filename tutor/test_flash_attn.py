from flash_attn import flash_attn_func
import torch
import math

def ref_impl(q, k, v):
    q = q.transpose(1, 2)
    k = k.transpose(1, 2)
    v = v.transpose(1, 2)

    out = (q @ k.transpose(-1, -2) / math.sqrt(q.size(-1))).softmax(dim=-1) @ v
    return out.transpose(1, 2)

q, k, v = (torch.randn(32, 1024, 12, 64, device="cuda", dtype=torch.bfloat16) for _ in range(3))
act = flash_attn_func(q, k, v)
ref = ref_impl(q, k, v)
assert torch.allclose(ref, act, atol=1e-2, rtol=1e-2), f"{ref=}\n{act=}"
print("pass")
