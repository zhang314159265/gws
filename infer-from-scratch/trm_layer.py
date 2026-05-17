import torch
from torch import nn

from attn import Attention
from ffn import FeedForward

class TransformerLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attention_norm = nn.RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.ffn_norm = nn.RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.attention = Attention(config)
        self.feed_forward = FeedForward(config)

    def forward(self, x, start_pos):
        h = x + self.attention(self.attention_norm(x), start_pos)
        return h + self.feed_forward(self.ffn_norm(h))
