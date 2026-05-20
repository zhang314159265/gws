import torch


class Rope:
    cos_cache = None
    sin_cache = None

    @classmethod
    def precompute_cis(cls, config):
        head_dim = config.head_dim
        half_dim = head_dim // 2

        positions = torch.arange(0, config.max_position_embeddings)
        inv_freq = 1.0 / (config.rope_theta ** (torch.arange(0, half_dim).float() / half_dim))
        freqs = torch.outer(positions, inv_freq)
        emb = torch.cat([freqs, freqs], dim=-1)
        cls.cos_cache = emb.cos()
        cls.sin_cache = emb.sin()

    @classmethod
    def rotate_half(cls, x):
        half = x.shape[-1] // 2
        x1 = x[..., :half]
        x2 = x[..., half:]
        return torch.cat([-x2, x1], dim=-1)

    @classmethod
    def apply(cls, qk, start_pos):
        seqlen = qk.shape[0]
        cos = cls.cos_cache[start_pos : start_pos + seqlen].to(qk.dtype)
        sin = cls.sin_cache[start_pos : start_pos + seqlen].to(qk.dtype)
        return qk * cos[:, None, :] + cls.rotate_half(qk) * sin[:, None, :]
