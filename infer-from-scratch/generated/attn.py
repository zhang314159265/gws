import torch
from torch import nn
import math

from rope import Rope
from rmsnorm import RMSNorm


class Attention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.head_dim = config.head_dim
        self.num_q_heads = config.num_q_heads
        self.num_kv_heads = config.num_kv_heads

        self.q_dim = self.head_dim * config.num_q_heads
        self.q_proj = nn.Linear(config.hidden_size, self.q_dim, bias=False)
        self.k_proj = nn.Linear(config.hidden_size, self.head_dim * config.num_kv_heads, bias=False)
        self.v_proj = nn.Linear(config.hidden_size, self.head_dim * config.num_kv_heads, bias=False)
        self.o_proj = nn.Linear(self.q_dim, config.hidden_size, bias=False)

        self.q_norm = RMSNorm(self.head_dim, config.rms_norm_eps)
        self.k_norm = RMSNorm(self.head_dim, config.rms_norm_eps)

        self.cache = torch.empty(2, config.max_position_embeddings, config.num_kv_heads, self.head_dim)
        self.temperature = math.sqrt(self.head_dim)

    def expand_kv(self, kv, g):
        seqlen, num_head, head_dim = kv.shape
        return kv[:, :, None, :].expand(-1, -1, g, -1).reshape(seqlen, num_head * g, head_dim)

    def apply_mask(self, score, start_pos):
        _, q_seqlen, kv_seqlen = score.shape
        assert kv_seqlen - q_seqlen == start_pos
        if q_seqlen == 1:
            return score
        mask = torch.full([kv_seqlen, kv_seqlen], float("-inf"), device=score.device)
        mask = torch.triu(mask, diagonal=1)
        mask = mask[start_pos : start_pos + q_seqlen]
        return score + mask

    def forward(self, x, start_pos):
        seqlen, hidden_size = x.shape
        q = self.q_proj(x).view(seqlen, self.num_q_heads, self.head_dim)
        k = self.k_proj(x).view(seqlen, self.num_kv_heads, self.head_dim)
        v = self.v_proj(x).view(seqlen, self.num_kv_heads, self.head_dim)

        q = self.q_norm(q)
        k = self.k_norm(k)

        q = Rope.apply(q, start_pos)
        k = Rope.apply(k, start_pos)

        self.cache[0, start_pos : start_pos + seqlen, :, :] = k
        self.cache[1, start_pos : start_pos + seqlen, :, :] = v
        k = self.cache[0, : start_pos + seqlen, :, :]
        v = self.cache[1, : start_pos + seqlen, :, :]

        g = self.num_q_heads // self.num_kv_heads
        k = self.expand_kv(k, g)
        v = self.expand_kv(v, g)

        q = q.transpose(0, 1)
        k = k.transpose(0, 1)
        v = v.transpose(0, 1)

        score = q @ k.transpose(-1, -2)
        score = score / self.temperature
        score = self.apply_mask(score, start_pos)
        score = torch.softmax(score, dim=-1, dtype=torch.float32).to(q.dtype)

        out = (score @ v).transpose(0, 1).reshape(seqlen, self.q_dim)
        return self.o_proj(out)
