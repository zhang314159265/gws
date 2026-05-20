import torch
from torch import nn
import torch.nn.functional as F

from ffn import Expert


class MoEFeedForward(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.gate = nn.Linear(config.hidden_size, config.num_experts, bias=False)
        self.experts = nn.ModuleList([
            Expert(config)
            for _ in range(config.num_experts)
        ])
        self.num_experts_per_tok = config.num_experts_per_tok
        self.norm_topk_prob = config.norm_topk_prob

    def forward(self, x):
        seqlen, hidden_size = x.shape
        router_logits = self.gate(x)
        routing_weights = F.softmax(router_logits, dim=-1, dtype=torch.float32)
        topk_weights, topk_indices = torch.topk(routing_weights, self.num_experts_per_tok, dim=-1)
        if self.norm_topk_prob:
            topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)
        topk_weights = topk_weights.to(x.dtype) # TODO why these type conversion?

        out = torch.zeros_like(x)
        for i in range(self.num_experts_per_tok):
            expert_indices = topk_indices[:, i]
            weights = topk_weights[:, i]
            for expert_id in expert_indices.unique():
                mask = expert_indices == expert_id
                expert_out = self.experts[expert_id](x[mask])
                out[mask] += weights[mask, None] * expert_out
        return out
