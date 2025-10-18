import torch
import torch.nn.functional as F

# @torch.compile
def ref_attn_fwd(Q, K, V):
    return F.scaled_dot_product_attention(Q, K, V, is_causal=True)

def ref_attn_bwd(dY, Q, K, V, Y):
    dQ, dK, dV = torch.autograd.grad(Y, [Q, K, V], dY, retain_graph=True)

    return dQ, dK, dV
    
