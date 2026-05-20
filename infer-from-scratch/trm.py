from torch import nn
from trm_layer import TransformerLayer
from rmsnorm import RMSNorm


class Transformer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList([
            TransformerLayer(config) for _ in range(config.num_layers)
        ])
        self.norm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    def forward(self, x, start_pos):
        x = self.embed_tokens(x)
        for layer in self.layers:
            x = layer(x, start_pos)
        x = self.norm(x)
        return self.lm_head(x[-1])
