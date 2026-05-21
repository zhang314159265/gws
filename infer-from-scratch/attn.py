import torch
from torch import nn

from rope import Rope
import math

class Attention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.head_dim = head_dim = config.head_dim
        self.q_proj = nn.Linear(config.hidden_size, head_dim * config.num_q_heads, bias=False)
        self.k_proj = nn.Linear(config.hidden_size, head_dim * config.num_kv_heads, bias=False)
        self.v_proj = nn.Linear(config.hidden_size, head_dim * config.num_kv_heads, bias=False)
        self.o_proj = nn.Linear(head_dim * config.num_q_heads, config.hidden_size, bias=False)
        self.cache = torch.empty(2, config.max_position_embeddings, config.num_kv_heads, head_dim)

        self.q_norm = nn.RMSNorm(head_dim, eps=config.rms_norm_eps)
        self.k_norm = nn.RMSNorm(head_dim, eps=config.rms_norm_eps)

        self.temperature = math.sqrt(self.head_dim)

    def expand_kv(self, kv, g):
        seqlen, num_head, head_dim = kv.shape
        return kv[:, :, None, :].expand(-1, -1, g, -1).reshape(seqlen, num_head * g, head_dim)

    def apply_mask_trivial(self, score, start_pos):
        _, q_seqlen, kv_seqlen = score.shape
        assert kv_seqlen - q_seqlen == start_pos
        if q_seqlen == 1:
            return score
        mask = torch.full([kv_seqlen, kv_seqlen], float("-inf"))
        mask = torch.triu(mask, diagonal=1)
        mask = mask[start_pos: start_pos + q_seqlen]
        return score + mask

    def apply_mask_triton(self, score, start_pos):
        _, q_seqlen, kv_seqlen = score.shape
        assert kv_seqlen - q_seqlen == start_pos
        if q_seqlen == 1:
            return score

        raise NotImplementedError("Triton version not implemented yet")

    apply_mask = apply_mask_trivial

    def forward(self, x, start_pos):
        seqlen, hidden_size = x.shape
        q = self.q_proj(x).view(seqlen, -1, self.head_dim)
        k = self.k_proj(x).view(seqlen, -1, self.head_dim)
        v = self.v_proj(x).view(seqlen, -1, self.head_dim)

        # q/k norm
        q = self.q_norm(q)
        k = self.k_norm(k)

        # apply rope
        q = Rope.apply(q, start_pos)
        k = Rope.apply(k, start_pos)

        # update kv cache
        self.cache[0, start_pos: start_pos + seqlen, :, :] = k
        self.cache[1, start_pos: start_pos + seqlen, :, :] = v
        k = self.cache[0, :start_pos + seqlen, :, :]
        v = self.cache[1, :start_pos + seqlen, :, :]

        # expand kv
        g = q.size(1) // k.size(1)
        k = self.expand_kv(k, g)
        v = self.expand_kv(v, g)

        # transpose qkv
        q = q.transpose(0, 1)
        k = k.transpose(0, 1)
        v = v.transpose(0, 1)

        # compute score
        score = q @ k.transpose(-1, -2)

        # apply temperature
        score = score / self.temperature

        # apply mask
        score = self.apply_mask(score, start_pos)

        # do softmax
        score = torch.softmax(score, dim=-1)

        # compute output
        out = (score @ v).transpose(0, 1).reshape(seqlen, -1)
        return self.o_proj(out)
