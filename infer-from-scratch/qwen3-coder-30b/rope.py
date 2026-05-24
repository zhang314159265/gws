import torch

"""
Qwen rope uses a different convention compared to llama.
For llama, q tensor is interpreted as:
    real0, imag0, real1, imag1 ...
For Qwen, q tensor is interpreted as:
    real0, real1, ... imag0, imag1, ...
"""

class Rope:
    cos = None
    sin = None

    @classmethod
    def precompute_cis(cls, config):
        head_dim = config.head_dim
        half_head = head_dim // 2

        a = torch.arange(0, config.max_position_embeddings)
        b = torch.pow(config.rope_theta, -torch.arange(0, half_head).float() / half_head)
        outer = torch.outer(a, b)
        outer = torch.cat([outer, outer], dim=-1)
        cls.cos = outer.cos().to(torch.bfloat16)
        cls.sin = outer.sin().to(torch.bfloat16)

    @classmethod
    def rotate(cls, qk):
        head_dim = qk.size(-1)
        half_head = head_dim // 2

        return torch.cat([
            -qk[:, :, half_head:],
            qk[:, :, :half_head],
        ], dim=-1)

    @classmethod
    def apply(cls, qk, start_pos):
        seqlen, _, _ = qk.shape

        if seqlen > 1:
            cos = cls.cos[start_pos: start_pos + seqlen, None, :]
            sin = cls.sin[start_pos: start_pos + seqlen, None, :]
        else:
            cos = cls.cos[:, None, :].index_select(0, start_pos)
            sin = cls.sin[:, None, :].index_select(0, start_pos)

        return qk * cos + cls.rotate(qk) * sin
