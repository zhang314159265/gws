import torch

class Rope:
    cis = None

    @classmethod
    def precompute_cis(cls, config):
        head_dim = config.hidden_size // config.num_q_heads
        half_head_dim = head_dim // 2

        a = torch.arange(0, config.max_position_embeddings)
        b = torch.pow(config.rope_theta, -torch.arange(0, half_head_dim) / half_head_dim)
        outer = torch.outer(a, b)
        cos = outer.cos()
        sin = outer.sin()
        cls.cis = torch.stack([cos, sin], dim=2).flatten(1)

    @classmethod
    def apply(cls, qk, start_pos):
        """
        Assume no batch dimension (i.e. batch-size == 1).
        """
        seqlen, num_head, head_dim = qk.shape
        cis = cls.cis[start_pos : start_pos + seqlen, :]
        qk = qk.transpose(0, 1)
        real = qk[..., ::2] * cis[..., ::2] - qk[..., 1::2] * cis[..., 1::2]
        imag = qk[..., ::2] * cis[..., 1::2] + qk[..., 1::2] * cis[..., ::2]
        out = torch.stack([real, imag], dim=3)
        out = out.flatten(2)
        out = out.transpose(0, 1)
        return out
