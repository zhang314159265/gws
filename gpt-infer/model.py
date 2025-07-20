from torch import nn
from dataclasses import dataclass
from typing import Optional
import torch
import torch.nn.functional as F
import math

LLAMA_3_1_CONFIG = dict(
    block_size=131072,
    vocab_size=128256,
    n_layer=32,
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

def find_multiple(n, k):
    if n % k == 0:
        return n
    return n - n % k + k

@dataclass
class ModelArgs:
    block_size: int
    vocab_size: int
    n_layer: int
    n_head: int
    dim: int
    intermediate_size: int
    n_local_heads: int
    rope_base: int
    rope_scaling: Optional[dict]
    norm_eps: float = 1e-5

    def __post_init__(self):
        self.head_dim = self.dim // self.n_head

class KVCache(nn.Module):
    def __init__(self, max_batch_size, max_seq_length, n_heads, head_dim, dtype=torch.bfloat16):
        super().__init__()
        cache_shape = (max_batch_size, n_heads, max_seq_length, head_dim)
        self.register_buffer("k_cache", torch.zeros(cache_shape, dtype=dtype))
        self.register_buffer("v_cache", torch.zeros(cache_shape, dtype=dtype))

    def update(self, input_pos, k_val, v_val):
        k_out = self.k_cache
        v_out = self.v_cache
        k_out[:, :, input_pos] = k_val
        v_out[:, :, input_pos] = v_val
        return k_out, v_out

class Attention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        total_head_dim = (config.n_head + 2 * config.n_local_heads) * config.head_dim
        self.wqkv = nn.Linear(config.dim, total_head_dim, bias=False)
        self.wo = nn.Linear(config.dim, config.dim, bias=False)
        self.kv_cache = None

    def attention(self, q, k, v, input_pos, mask_cache):
        return F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=mask_cache,
            enable_gqa=self.config.n_head != self.config.n_local_heads
        )

    def forward(self, x, freqs_cis, input_pos, mask_cache):
        bsz, seqlen, _ = x.shape

        kv_size = self.config.n_local_heads * self.config.head_dim
        q, k, v = self.wqkv(x).split([self.config.dim, kv_size, kv_size], dim=-1)

        q = q.view(bsz, seqlen, self.config.n_head, self.config.head_dim)
        k = k.view(bsz, seqlen, self.config.n_local_heads, self.config.head_dim)
        v = v.view(bsz, seqlen, self.config.n_local_heads, self.config.head_dim)

        q = apply_rotary_emb(q, freqs_cis)
        k = apply_rotary_emb(k, freqs_cis)

        q, k, v = map(lambda x: x.transpose(1, 2), (q, k, v))

        if self.kv_cache is not None:
            k, v = self.kv_cache.update(input_pos, k, v)

        y = self.attention(q, k, v, input_pos, mask_cache)

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
        self.attention = Attention(config)
        self.feed_forward = FeedForward(config)
        self.ffn_norm = RMSNorm(config.dim, config.norm_eps)
        self.attention_norm = RMSNorm(config.dim, config.norm_eps)

    def forward(self, x, input_pos, freqs_cis, mask_cache):
        h = x + self.attention(self.attention_norm(x), freqs_cis, input_pos, mask_cache)
        out = h + self.feed_forward(self.ffn_norm(h))
        return out

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

def precompute_freqs_cis(
    seq_len, n_elem, base,
    dtype, rope_scaling
):
    assert n_elem % 2 == 0
    freqs = base ** -(torch.arange(0, n_elem, 2).float() / n_elem)
    if rope_scaling is not None:
        freqs = apply_rope_scaling(freqs, rope_scaling)
    t = torch.arange(seq_len, device=freqs.device)
    freqs = torch.outer(t, freqs)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
    cache = torch.stack([freqs_cis.real, freqs_cis.imag], dim=-1)
    return cache.to(dtype=dtype)

class Transformer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.tok_embeddings = nn.Embedding(config.vocab_size, config.dim)
        self.layers = nn.ModuleList(TransformerBlock(config) for _ in range(config.n_layer))
        self.norm = RMSNorm(config.dim, eps=config.norm_eps)
        self.output = nn.Linear(config.dim, config.vocab_size, bias=False)
        self.max_batch_size = -1
        self.max_seq_length = -1
        self.freqs_cis = None
        self.mask_cache = None

    def setup_caches(self, max_batch_size, max_seq_length):
        if self.max_seq_length >= max_seq_length and self.max_batch_size >= max_batch_size:
            return
        max_seq_length = find_multiple(max_seq_length, 8)
        self.max_batch_size = max_batch_size
        self.max_seq_length = max_seq_length
        dtype = self.output.weight.dtype
        for b in self.layers:
            b.attention.kv_cache = KVCache(max_batch_size, max_seq_length, self.config.n_local_heads, self.config.head_dim, dtype)

        self.freqs_cis = precompute_freqs_cis(self.config.block_size, self.config.head_dim, self.config.rope_base, dtype, self.config.rope_scaling)

        self.mask_cache = torch.ones(max_seq_length, max_seq_length, dtype=dtype).tril() != 0

    def forward(self, idx, input_pos):
        assert self.freqs_cis is not None

        freqs_cis = self.freqs_cis[input_pos]
        x = self.tok_embeddings(idx)

        for layer in self.layers:
            x = layer(x, input_pos, freqs_cis, self.mask_cache[input_pos])
        x = self.norm(x)
        logits = self.output(x)
        return logits

    @classmethod
    def create(cls):
        return cls(ModelArgs(**LLAMA_3_1_CONFIG))

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
