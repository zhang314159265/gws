from torch import nn

from attn import Attention
from ffn import MoEFeedForward
from torch.nn import RMSNorm


class TransformerLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.input_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.self_attn = Attention(config)
        self.mlp = MoEFeedForward(config)

    def forward(self, x, start_pos):
        h = x + self.self_attn(self.input_layernorm(x), start_pos)
        return h + self.mlp(self.post_attention_layernorm(h))
