"""
Benchmark torch.compile by compiling a single Llama3 TransformerBlock with fp8 training.
"""

import torch
from torch import nn
from dataclasses import dataclass
from typing import Optional
from torchao.float8 import convert_to_float8_training, Float8LinearConfig
import math
from torch.nn.attention import sdpa_kernel, SDPBackend
import torch.nn.functional as F

LLAMA_3_1_CONFIG = dict(
    n_head=32,
    dim=4096,
    intermediate_size=14336,
    n_local_heads=8,
    rope_base=500000,
    rope_scaling=dict(
        factor=8.0,
        low_freq_factor=1.0,
        high_freq_factor=4.0,
        original_max_position_embeddings=8192,
    ),
)

@dataclass
class ModelArgs:
    n_head: int
    dim: int
    intermediate_size: int
    n_local_heads: int
    rope_base: int
    rope_scaling: Optional[dict]
    norm_eps: float = 1e-5

    def __post_init__(self):
        self.head_dim = self.dim // self.n_head

class RMSNorm(nn.Module):
    def __init__(self, dim, eps):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(torch.mean(x * x, dim=-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        return output * self.weight

class Attention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        total_head_dim = (config.n_head + 2 * config.n_local_heads) * config.head_dim
        self.wqkv = nn.Linear(config.dim, total_head_dim, bias=False)
        self.wo = nn.Linear(config.dim, config.dim, bias=False)

    def attention(self, q, k, v):
        with sdpa_kernel([SDPBackend.FLASH_ATTENTION], set_priority=True):
            return F.scaled_dot_product_attention(
                q, k, v,
                is_causal=True,
            )

    @staticmethod
    def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
        bs, slen, n_kv_heads, head_dim = x.shape
        if n_rep == 1:
            return x
        return (
            torch.unsqueeze(x, dim=3)
            .expand(bs, slen, n_kv_heads, n_rep, head_dim)
            .reshape(bs, slen, n_kv_heads * n_rep, head_dim)
        )

    def forward(self, x, freqs_cis):
        bsz, seqlen, _ = x.shape

        kv_size = self.config.n_local_heads * self.config.head_dim
        q, k, v = self.wqkv(x).split([self.config.dim, kv_size, kv_size], dim=-1)

        q = q.view(bsz, seqlen, self.config.n_head, self.config.head_dim)
        k = k.view(bsz, seqlen, self.config.n_local_heads, self.config.head_dim)
        v = v.view(bsz, seqlen, self.config.n_local_heads, self.config.head_dim)

        q = apply_rotary_emb(q, freqs_cis)
        k = apply_rotary_emb(k, freqs_cis)

        factor  = self.config.n_head // self.config.n_local_heads
        k = self.repeat_kv(k, factor)
        v = self.repeat_kv(v, factor)

        q, k, v = map(lambda x: x.transpose(1, 2), (q, k, v))

        y = self.attention(q, k, v)

        y = y.transpose(1, 2).contiguous().view(bsz, seqlen, self.config.dim)

        y = self.wo(y)
        return y

class FeedForward(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.w1 = nn.Linear(config.dim, config.intermediate_size, bias=False)
        self.w3 = nn.Linear(config.dim, config.intermediate_size, bias=False)
        self.w2 = nn.Linear(config.intermediate_size, config.dim, bias=False)

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class TransformerBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.attention = Attention(config)
        self.feed_forward = FeedForward(config)
        self.ffn_norm = RMSNorm(config.dim, config.norm_eps)
        self.attention_norm = RMSNorm(config.dim, config.norm_eps)

        self.freqs_cis = None

    def forward(self, x):
        h = x + self.attention(self.attention_norm(x), self.freqs_cis)
        out = h + self.feed_forward(self.ffn_norm(h))
        return out

    def setup_caches(self, max_batch_size, max_seq_length):
        max_seq_length = find_multiple(max_seq_length, 8)
        dtype = self.feed_forward.w1.weight.dtype
        self.freqs_cis = precompute_freqs_cis(max_seq_length, self.config.head_dim, self.config.rope_base, dtype, self.config.rope_scaling)

    @classmethod
    def create(cls):
        return cls(ModelArgs(**LLAMA_3_1_CONFIG))

def _load_model(device, precision):
    model = TransformerBlock.create()
    model = model.to(device=device, dtype=precision)
    return model

def main():
    device = "cuda"
    x = torch.randn([1, 8192, 4096], device="cuda", dtype=torch.bfloat16, requires_grad=True)
    grad_y = torch.randn_like(x).requires_grad_(False)
    precision = torch.bfloat16

    block = _load_model(device, precision)

    def module_filter_fn(mod: torch.nn.Module, fqn: str):
        return True

    config = Float8LinearConfig.from_recipe_name("rowwise")
    convert_to_float8_training(block, config=config, module_filter_fn=module_filter_fn)

    with torch.device(device):
        block.setup_caches(max_batch_size=1, max_seq_length=8192)

    cblock = torch.compile(block, fullgraph=True)

    def f():
        cblock.zero_grad(True)
        x.grad = None
        y = cblock(x)
        y.backward(grad_y, retain_graph=True)

    for _ in range(3):
        f()

    from triton.testing import do_bench
    ms = do_bench(f, rep=50 * 20)
    print(f"ms={ms:.3f}")
    print("bye")
    exit() 

def find_multiple(n, k):
    if n % k == 0:
        return n
    return n - n % k + k

def precompute_freqs_cis(seq_len, n_elem, base, dtype, rope_scaling):
    assert n_elem % 2 == 0
    freqs = base ** -(torch.arange(0, n_elem, 2).float() / n_elem)
    if rope_scaling is not None:
        freqs = apply_rope_scaling(freqs, rope_scaling)
    t = torch.arange(seq_len, device=freqs.device)
    freqs = torch.outer(t, freqs)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
    cache = torch.stack([freqs_cis.real, freqs_cis.imag], dim=-1)
    return cache.to(dtype=dtype)

def apply_rope_scaling(freqs, rope_scaling):
    assert isinstance(rope_scaling, dict)
    factor = rope_scaling["factor"]
    low_freq_factor = rope_scaling["low_freq_factor"]
    high_freq_factor = rope_scaling["high_freq_factor"]
    old_context_len = rope_scaling["original_max_position_embeddings"]

    low_freq_wavelen = old_context_len / low_freq_factor
    high_freq_wavelen = old_context_len / high_freq_factor

    assert low_freq_wavelen > high_freq_wavelen
    new_freqs = []
    for freq in freqs:
        wavelen = 2 * math.pi / freq
        if wavelen < high_freq_wavelen:
            new_freqs.append(freq)
        elif wavelen > low_freq_wavelen:
            new_freqs.append(freq / factor)
        else:
            smooth = (old_context_len / wavelen - low_freq_factor) / (high_freq_factor - low_freq_factor)
            new_freqs.append((1 - smooth) * freq / factor + smooth * freq)
    return torch.tensor(new_freqs, dtype=freqs.dtype, device=freqs.device)

def apply_rotary_emb(x, freqs_cis):
    xshaped = x.float().reshape(*x.shape[:-1], -1, 2)
    freqs_cis = freqs_cis.view(1, xshaped.size(1), 1, xshaped.size(3), 2)
    x_out = torch.stack(
        [
            xshaped[..., 0] * freqs_cis[..., 0] - xshaped[..., 1] * freqs_cis[..., 1],
            xshaped[..., 0] * freqs_cis[..., 1] + xshaped[..., 1] * freqs_cis[..., 0],
        ],
        -1,
    )
    x_out = x_out.flatten(3)
    return x_out.type_as(x)

if __name__ == "__main__":
    main()
