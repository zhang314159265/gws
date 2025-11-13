import torch
import torch.nn.functional as F
from torch.nn.attention.flex_attention import flex_attention, create_block_mask
import math
import os

COMPILE = os.environ.get("COMPILE", "1") == "1"
USE_FLEX_DECODING = os.environ.get("USE_FLEX_DECODING", "1") == "1"

if COMPILE:
    flex_attention = torch.compile(flex_attention)

def manual_attn(q, k, v):
    if q.size(1) != k.size(1):
        assert q.size(1) % k.size(1) == 0
        ratio = q.size(1) // k.size(1)
        B, _, S, D = k.shape
        k = k[:, :, None, :, :].expand(-1, -1, ratio, -1, -1).reshape(B, -1, S, D)
        v = v[:, :, None, :, :].expand(-1, -1, ratio, -1, -1).reshape(B, -1, S, D)
    score = q @ k.transpose(-1, -2)

    # scale
    score /= math.sqrt(q.shape[-1])

    # mask
    S1, S2 = q.shape[-2], k.shape[-2]
    mask = torch.ones(S1, S2)
    mask = torch.tril(mask)
    score = torch.where(mask == 1, score, float('-inf'))

    score = torch.softmax(score, dim=-1)
    return score @ v

def flex_attn_score_mod(q, k, v):
    def score_mod(score, b_idx, h_idx, q_idx, kv_idx):
        return torch.where(q_idx >= kv_idx, score, float("-inf"))
    return flex_attention(q, k, v, score_mod=score_mod, enable_gqa=q.size(1) != k.size(1))

def flex_attn_mask_mod(q, k, v):
    def mask_mod(b_idx, h_idx, q_idx, kv_idx):
        return q_idx >= kv_idx
    S1, S2 = q.shape[-2], k.shape[-2]
    block_mask = create_block_mask(mask_mod, 1, 1, S1, S2)
    return flex_attention(q, k, v, block_mask=block_mask, enable_gqa=q.size(1) != k.size(1))

def numeric_check(ref, act):
    correct = torch.allclose(ref, act, atol=1e-3, rtol=1e-3)
    if not correct:
        breakpoint()
        assert False, "numeric check fails"

def main():
    B, H, S, D = 2, 4, 1024, 64
    q, k, v = [torch.randn(B, H, S, D) for _ in range(3)]
    if USE_FLEX_DECODING:
        q = torch.randn(B, H * 2, 4, D)

    ref = manual_attn(q, k, v)
    numeric_check(ref, F.scaled_dot_product_attention(q, k, v, is_causal=True, enable_gqa=q.size(1) != k.size(1)))
    numeric_check(ref, flex_attn_score_mod(q, k, v))
    numeric_check(ref, flex_attn_mask_mod(q, k, v))
    print("pass")

with torch.device("cuda"):
    main()
