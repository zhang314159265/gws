import torch
from torch import nn
from trm_layer import TransformerLayer

class Transformer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.tok_embeddings = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList([
            TransformerLayer(config) for _ in range(config.num_layers)
        ])
        self.norm = nn.RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.output = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    def forward(self, x, start_pos):
        x = self.tok_embeddings(x)
        for layer in self.layers:
            x = layer(x, start_pos)
        x = self.norm(x)
        out = self.output(x[-1])
        return out
