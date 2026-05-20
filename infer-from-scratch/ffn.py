import torch
from torch import nn

class Expert(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.gate_proj = nn.Linear(config.hidden_size, config.moe_intermediate_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, config.moe_intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.moe_intermediate_size, config.hidden_size, bias=False)

    def forward(self, x):
        gate = self.gate_proj(x)
        up = self.up_proj(x)
        return self.down_proj(gate * gate.sigmoid() * up)

class MoEFeedForward(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.gate = nn.Linear(config.hidden_size, config.num_experts, bias=False)
        self.experts = nn.ModuleList([
            Expert(config) for _ in range(config.num_experts)
        ])
        self.num_experts_per_tok = config.num_experts_per_tok
        self.norm_topk_prob = config.norm_topk_prob

    def forward(self, x):
        routing_logits = self.gate(x)
        routing_weights = torch.softmax(routing_logits, dim=-1, dtype=torch.float32)

        topk_weights, topk_indices = torch.topk(routing_weights, self.num_experts_per_tok, dim=-1)
        if self.norm_topk_prob:
            topk_weights /= topk_weights.sum(dim=-1, keepdim=True)
        topk_weights = topk_weights.to(x.dtype)

        out = torch.zeros_like(x)
        for slot in range(self.num_experts_per_tok):
            weights = topk_weights[:, slot]
            indices = topk_indices[:, slot]
            for expert_id in indices.unique():
                mask = indices == expert_id
                expert_out = self.experts[expert_id](x[mask])
                out[mask] += expert_out * weights[mask, None]
        return out
