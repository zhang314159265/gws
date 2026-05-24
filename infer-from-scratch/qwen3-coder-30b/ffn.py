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

    def _group_expert_weights(self):
        # TODO: avoid double the MoE memory usage!
        self.gate_weights = torch.stack([expert.gate_proj.weight for expert in self.experts], dim=0)
        self.up_weights = torch.stack([expert.up_proj.weight for expert in self.experts], dim=0)
        self.down_weights = torch.stack([expert.down_proj.weight for expert in self.experts], dim=0)

    def forward_unique(self, x):
        """
        Does not work with CG due to the using of torch.unique
        """
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

    def forward_bmm(self, x):
        # [S, M]
        seqlen, hidden_size = x.shape
        routing_logits = self.gate(x)
        routing_weights = torch.softmax(routing_logits, dim=-1, dtype=torch.float32)

        # [S, A]
        topk_weights, topk_indices = torch.topk(routing_weights, self.num_experts_per_tok, dim=-1)
        if self.norm_topk_prob:
            topk_weights /= topk_weights.sum(dim=-1, keepdim=True)
        topk_weights = topk_weights.to(x.dtype)

        # [S, A, N, M]
        gate_weights = self.gate_weights[topk_indices].flatten(0, 1).transpose(-1, -2)
        up_weights = self.up_weights[topk_indices].flatten(0, 1).transpose(-1, -2)
        down_weights = self.down_weights[topk_indices].flatten(0, 1).transpose(-1, -2)

        x_expanded = x[:, None, None, :].expand(-1, self.num_experts_per_tok, -1, -1).flatten(0, 1)

        # [SA, 1, N]
        gate = torch.bmm(x_expanded, gate_weights)
        up = torch.bmm(x_expanded, up_weights)
        # [S, A, N]
        y = torch.bmm(gate * gate.sigmoid() * up, down_weights).unsqueeze(1).view(seqlen, self.num_experts_per_tok, -1)
        return (y * topk_weights[..., None]).sum(dim=1)

    def forward_einsum(self, x):
        # [S, M]
        seqlen, hidden_size = x.shape
        routing_logits = self.gate(x)
        routing_weights = torch.softmax(routing_logits, dim=-1, dtype=torch.float32)

        # [S, A]
        topk_weights, topk_indices = torch.topk(routing_weights, self.num_experts_per_tok, dim=-1)
        if self.norm_topk_prob:
            topk_weights /= topk_weights.sum(dim=-1, keepdim=True)
        topk_weights = topk_weights.to(x.dtype)

        # [S, A, N, M]
        gate_weights = self.gate_weights[topk_indices]
        up_weights = self.up_weights[topk_indices]
        down_weights = self.down_weights[topk_indices]

        gate = torch.einsum("SM,SANM->SAN", x, gate_weights)
        up = torch.einsum("SM,SANM->SAN", x, up_weights)
        y = gate * gate.sigmoid() * up
        y = torch.einsum("SAM,SANM->SAN", y, down_weights)
        y = (y * topk_weights[..., None]).sum(dim=1)
        return y

    forward = forward_einsum
