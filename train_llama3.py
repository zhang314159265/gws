"""
Train llama3.1 8B.
"""

import time
import os
from dataclasses import dataclass
from torch import nn
from torch import Tensor
import math
import torch.nn.functional as F
import torch
from typing import Optional
from torch.nn.attention.flex_attention import (
    _mask_mod_signature,
    BlockMask,
    create_block_mask,
    flex_attention,
)

# BEGIN code copied from gpt-fast

def causal_mask(b, h, q, kv):
    return q >= kv

def find_multiple(n: int, k: int) -> int:
    if n % k == 0:
        return n
    return n + k - (n % k)




def get_mask_mod(mask_mod: _mask_mod_signature, offset: int):
    def _mask_mod(b, h, q, kv):
        return mask_mod(b, h, q + offset, kv)

    return _mask_mod


NLAYER = int(os.getenv("NLAYER", 32))
print(f"#layer {NLAYER}")
transformer_configs = {
    "llama-3.1-8b": dict(block_size=131072, n_layer=NLAYER, n_head=32, n_local_heads=8, dim=4096, intermediate_size=14336, vocab_size=128256, rope_base=500000,
        rope_scaling=dict(factor=8.0, low_freq_factor=1.0, high_freq_factor=4.0, original_max_position_embeddings=8192),
    ),
}

@dataclass
class ModelArgs:
    block_size: int = 2048
    vocab_size: int = 32000
    n_layer: int = 32
    n_head: int = 32
    dim: int = 4096
    intermediate_size: int = None
    n_local_heads: int = -1
    head_dim: int = 64
    rope_base: float = 10000
    norm_eps: float = 1e-5
    rope_scaling: Optional[dict] = None

    def __post_init__(self):
        if self.n_local_heads == -1:
            self.n_local_heads = self.n_head
        if self.intermediate_size is None:
            hidden_dim = 4 * self.dim
            n_hidden = int(2 * hidden_dim / 3)
            self.intermediate_size = find_multiple(n_hidden, 256)
        self.head_dim = self.dim // self.n_head

    @classmethod
    def from_name(cls, name: str):
        if name in transformer_configs:
            return cls(**transformer_configs[name])
        # fuzzy search
        config = [config for config in transformer_configs if config.lower() in str(name).lower()]

        # We may have two or more configs matched (e.g. "7B" and "Mistral-7B"). Find the best config match,
        # take longer name (as it have more symbols matched)
        if len(config) > 1:
            config.sort(key=len, reverse=True)
            assert len(config[0]) != len(config[1]), name # make sure only one 'best' match
            
        return cls(**transformer_configs[config[0]])

class KVCache(nn.Module):
    def __init__(self, max_batch_size, max_seq_length, n_heads, head_dim, dtype=torch.bfloat16):
        super().__init__()
        cache_shape = (max_batch_size, n_heads, max_seq_length, head_dim)
        self.register_buffer('k_cache', torch.zeros(cache_shape, dtype=dtype))
        self.register_buffer('v_cache', torch.zeros(cache_shape, dtype=dtype))

    def update(self, input_pos, k_val, v_val):
        # input_pos: [S], k_val: [B, H, S, D]
        assert input_pos.shape[0] == k_val.shape[2]

        k_out = self.k_cache
        v_out = self.v_cache
        k_out[:, :, input_pos] = k_val
        v_out[:, :, input_pos] = v_val

        return k_out, v_out

class Transformer(nn.Module):
    def __init__(self, config: ModelArgs) -> None:
        super().__init__()
        self.config = config

        self.tok_embeddings = nn.Embedding(config.vocab_size, config.dim)
        self.layers = nn.ModuleList(TransformerBlock(config) for _ in range(config.n_layer))
        self.norm = RMSNorm(config.dim, eps=config.norm_eps)
        self.output = nn.Linear(config.dim, config.vocab_size, bias=False)

        self.freqs_cis: Optional[Tensor] = None
        self.mask_cache: Optional[Tensor] = None
        self.max_batch_size = -1
        self.max_seq_length = -1
        self.get_mask_mod = get_mask_mod

    def setup_caches(self, max_batch_size, max_seq_length):
        if self.max_seq_length >= max_seq_length and self.max_batch_size >= max_batch_size:
            return
        head_dim = self.config.dim // self.config.n_head
        max_seq_length = find_multiple(max_seq_length, 8)
        self.max_seq_length = max_seq_length
        self.max_batch_size = max_batch_size
        dtype = self.output.weight.dtype
        # For quantized layers, dtype is encoded in scales
        if hasattr(self.output, "scales"):
            dtype = self.output.scales.dtype
        elif hasattr(self.output, "scales_and_zeros"):
            dtype = self.output.scales_and_zeros.dtype
        # We don't need KVCache for training
        # for b in self.layers:
        #     b.attention.kv_cache = KVCache(max_batch_size, max_seq_length, self.config.n_local_heads, head_dim, dtype)

        self.freqs_cis = precompute_freqs_cis(self.config.block_size, self.config.dim // self.config.n_head, self.config.rope_base, dtype, self.config.rope_scaling)

    def forward(self, mask: BlockMask, idx: Tensor, input_pos: Optional[Tensor] = None) -> Tensor:
        assert self.freqs_cis is not None, "Caches must be initialized first"
        # mask.mask_mod = self.get_mask_mod(mask.mask_mod, input_pos[0])
        freqs_cis = self.freqs_cis[input_pos]
        x = self.tok_embeddings(idx)

        for i, layer in enumerate(self.layers):
            x = layer(x, input_pos, freqs_cis, mask)
        x = self.norm(x)
        logits = self.output(x)
        return logits

    @classmethod
    def from_name(cls, name: str):
        return cls(ModelArgs.from_name(name))


class TransformerBlock(nn.Module):
    def __init__(self, config: ModelArgs) -> None:
        super().__init__()
        self.attention = Attention(config)
        self.feed_forward = FeedForward(config)
        self.ffn_norm = RMSNorm(config.dim, config.norm_eps)
        self.attention_norm = RMSNorm(config.dim, config.norm_eps)

    def forward(self, x: Tensor, input_pos: Tensor, freqs_cis: Tensor, mask: BlockMask) -> Tensor:
        h = x + self.attention(self.attention_norm(x), freqs_cis, mask, input_pos)
        out = h + self.feed_forward(self.ffn_norm(h))
        return out


class Attention(nn.Module):
    def __init__(self, config: ModelArgs):
        super().__init__()
        assert config.dim % config.n_head == 0

        total_head_dim = (config.n_head + 2 * config.n_local_heads) * config.head_dim
        # key, query, value projections for all heads, but in a batch
        self.wqkv = nn.Linear(config.dim, total_head_dim, bias=False)
        self.wo = nn.Linear(config.dim, config.dim, bias=False)
        self.kv_cache = None

        self.n_head = config.n_head
        self.head_dim = config.head_dim
        self.n_local_heads = config.n_local_heads
        self.dim = config.dim
        self._register_load_state_dict_pre_hook(self.load_hook)

    def load_hook(self, state_dict, prefix, *args):
        if prefix + "wq.weight" in state_dict:
            wq = state_dict.pop(prefix + "wq.weight")
            wk = state_dict.pop(prefix + "wk.weight")
            wv = state_dict.pop(prefix + "wv.weight")
            state_dict[prefix + "wqkv.weight"] = torch.cat([wq, wk, wv])

    def forward(self, x: Tensor, freqs_cis: Tensor, mask: BlockMask, input_pos: Optional[Tensor] = None) -> Tensor:
        bsz, seqlen, _ = x.shape

        kv_size = self.n_local_heads * self.head_dim
        q, k, v = self.wqkv(x).split([self.dim, kv_size, kv_size], dim=-1)

        q = q.view(bsz, seqlen, self.n_head, self.head_dim)
        k = k.view(bsz, seqlen, self.n_local_heads, self.head_dim)
        v = v.view(bsz, seqlen, self.n_local_heads, self.head_dim)

        q = apply_rotary_emb(q, freqs_cis)
        k = apply_rotary_emb(k, freqs_cis)

        q, k, v = map(lambda x: x.transpose(1, 2), (q, k, v))

        if self.kv_cache is not None:
            k, v = self.kv_cache.update(input_pos, k, v)

        q_per_kv = self.n_head // self.n_local_heads
        expand_shape = (bsz, self.n_local_heads, q_per_kv, -1, self.head_dim)
        k = k.unsqueeze(2).expand(expand_shape).flatten(1, 2)
        v = v.unsqueeze(2).expand(expand_shape).flatten(1, 2)

        # y = flex_attention(q, k, v, block_mask=mask, enable_gqa=(self.n_head != self.n_local_heads))
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)

        y = y.transpose(1, 2).contiguous().view(bsz, seqlen, self.dim)

        y = self.wo(y)
        return y


class FeedForward(nn.Module):
    def __init__(self, config: ModelArgs) -> None:
        super().__init__()
        self.w1 = nn.Linear(config.dim, config.intermediate_size, bias=False)
        self.w3 = nn.Linear(config.dim, config.intermediate_size, bias=False)
        self.w2 = nn.Linear(config.intermediate_size, config.dim, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(torch.mean(x * x, dim=-1, keepdim=True) + self.eps)

    def forward(self, x: Tensor) -> Tensor:
        output = self._norm(x.float()).type_as(x)
        return output * self.weight


def apply_rope_scaling(freqs: torch.Tensor, rope_scaling: Optional[dict] = None):
    factor = rope_scaling["factor"]
    low_freq_factor = rope_scaling["low_freq_factor"]
    high_freq_factor = rope_scaling["high_freq_factor"]
    old_context_len = rope_scaling["original_max_position_embeddings"]

    low_freq_wavelen = old_context_len / low_freq_factor
    high_freq_wavelen = old_context_len / high_freq_factor
    new_freqs = []
    for freq in freqs:
        wavelen = 2 * math.pi / freq
        if wavelen < high_freq_wavelen:
            new_freqs.append(freq)
        elif wavelen > low_freq_wavelen:
            new_freqs.append(freq / factor)
        else:
            assert low_freq_wavelen != high_freq_wavelen
            smooth = (old_context_len / wavelen - low_freq_factor) / (high_freq_factor - low_freq_factor)
            new_freqs.append((1 - smooth) * freq / factor + smooth * freq)
    return torch.tensor(new_freqs, dtype=freqs.dtype, device=freqs.device)


def precompute_freqs_cis(
    seq_len: int, n_elem: int, base: int = 10000,
    dtype: torch.dtype = torch.bfloat16,
    rope_scaling: Optional[dict] = None,
) -> Tensor:
    freqs = 1.0 / (base ** (torch.arange(0, n_elem, 2)[: (n_elem // 2)].float() / n_elem))
    if rope_scaling is not None:
        freqs = apply_rope_scaling(freqs, rope_scaling)
    t = torch.arange(seq_len, device=freqs.device)
    freqs = torch.outer(t, freqs)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
    cache = torch.stack([freqs_cis.real, freqs_cis.imag], dim=-1)
    return cache.to(dtype=dtype)


def apply_rotary_emb(x: Tensor, freqs_cis: Tensor) -> Tensor:
    xshaped = x.float().reshape(*x.shape[:-1], -1, 2)
    freqs_cis = freqs_cis.view(1, xshaped.size(1), 1, xshaped.size(3), 2)
    x_out2 = torch.stack(
        [
            xshaped[..., 0] * freqs_cis[..., 0] - xshaped[..., 1] * freqs_cis[..., 1],
            xshaped[..., 1] * freqs_cis[..., 0] + xshaped[..., 0] * freqs_cis[..., 1],
        ],
        -1,
    )

    x_out2 = x_out2.flatten(3)
    return x_out2.type_as(x)

# END code copied from gpt-fast

def trace_ready(prof):
    path = "/tmp/chrome.json.gz"
    prof.export_chrome_trace(path)
    print(f"Profile written to {path}")

@torch.compile
def fwd_and_bwd(model, idx, input_pos, target):
    logits = model(mask, idx, input_pos)
    loss = F.cross_entropy(logits.view(-1, logits.size(-1)), target.view(-1))
    loss.backward()



if __name__ == "__main__":
    torch.set_default_device("cuda")
    ncu_profile_step = 7

    torch.randn(1)
    torch.cuda.cudart().cudaProfilerStop()

    model = Transformer.from_name("llama-3.1-8b").to(device="cuda", dtype=torch.bfloat16)
    # Non compile case
    # batch_size, seq_len = 128, 512
    # batch_size, seq_len = 2, 512 # 64G
    # batch_size, seq_len = 8, 512 # 80G
    # batch_size, seq_len = 16, 512 # OOM
    # batch_size, seq_len = 8, 128 # 66.97GB with torch.compile

    # with compile
    # batch_size, seq_len = 128, 512 # OOM
    batch_size, seq_len = 16, 512 # 96.6GB, 10K tokens/s, 822ms
    model.setup_caches(max_batch_size=batch_size, max_seq_length = seq_len)
    # model = torch.compile(model)

    create_block_mask = torch.compile(create_block_mask)
    mask = create_block_mask(causal_mask, batch_size, model.config.n_head, seq_len, seq_len)
    input_pos = torch.arange(0, seq_len)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, betas=(0.9, 0.95), weight_decay=0.0, fused=True)
    # optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, betas=(0.9, 0.95), weight_decay=0.0, capturable=True, foreach=True)

    profiler = torch.profiler.profile(
        schedule=torch.profiler.schedule(
            wait=2,
            warmup=2,
            active=3,
            repeat=1,
        ),
        on_trace_ready=trace_ready,
    )
    profiler.start()

    print(f"{ncu_profile_step=}")
    for step in range(15):
        if step == ncu_profile_step:
            torch.cuda.cudart().cudaProfilerStart()
        torch.cuda.synchronize()
        t0 = time.time()
        idx = torch.randint(0, model.config.vocab_size, (batch_size, seq_len), device="cuda")
        target = torch.randint(0, model.config.vocab_size, (batch_size, seq_len), device="cuda")
        fwd_and_bwd(model, idx, input_pos, target)
        torch.cuda.synchronize()

        # don't profile the optimizer
        torch.cuda.cudart().cudaProfilerStop()

        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        torch.cuda.synchronize()
        profiler.step()

        elapse = time.time() - t0
        tps = batch_size * seq_len / elapse

        print(f"Step {step}: {elapse * 1000:.3f}ms, {tps:.3f}tokens/s")

        if step == 3:
            peak_mem = torch.cuda.max_memory_allocated() / 10**9
            print(f"peak_mem: {peak_mem:.3f} GB")

    profiler.stop()

# if False: # old manual version
#     class TransformerLayer(nn.Module):
#         def __init__(self, config):
#             super().__init__()
#             self.norm_1 = nn.RMSNorm(config.embed_dim, config.norm_eps)
#             self.attn = None
#             self.norm_2 = nn.RMSNorm(config.embed_dim, config.norm_eps)
#             self.mlp = None
#     
#         def forward(self, x):
#             x = x + self.attn(self.norm_1(x))
#             x = x + self.mlp(self.norm_2(x))
#             return x
#     
#     @dataclass
#     class ModelConfig:
#         vocab_size = 128_256
#         num_layers = 32
#         num_heads = 32
#         num_kv_heads = 8
#         embed_dim = 4096
#         block_size = 1024
#         intermediate_dim = 14336
#         norm_eps = 1e-5
#     
#     class Llama(nn.Module):
#         def __init__(self, config):
#             super().__init__()
#     
#             self.tok_embeddings = nn.Embedding(config.vocab_size, config.embed_dim)
#             self.layers = nn.Sequential(*[
#                 TransformerLayer(config) for _ in range(config.num_layers)
#             ])
#             self.output_proj = nn.Linear(config.embed_dim, config.vocab_size, bias=False)
#     
#             self.final_norm = nn.RMSNorm(config.embed_dim, eps=config.norm_eps)
#     
#         def forward(self, idx, target):
#             x = self.tok_embeddings(idx)
#             x = self.layers(x)
#             x = self.final_norm(x)
#             x = self.output_proj(x)
#             loss = F.cross_entropy(x.view(-1, x.size(-1)), target.view(-1))
#             return loss
#     
#     if __name__ == "__main__":
#         config = ModelConfig()
#         model = Llama(config).to("cuda")
#     
#         batch_size = 128
#         seq_len = 512
#     
#         idx = torch.randint(0, config.vocab_size, (batch_size, seq_len), device="cuda")
#         target = torch.randint(0, config.vocab_size, (batch_size, seq_len), device="cuda")
#         loss = model(idx, target)
#         print(f"{loss=}")
#         
#         # print(model)
