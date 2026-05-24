import regex

def get_num_params(model):
    tot = 0

    for param in model.parameters():
        tot += param.numel()
    return tot

def get_num_active_params(model, config):
    num_experts = config.num_experts
    num_active_experts = config.num_experts_per_tok

    tot = 0
    expert_tot = 0

    pat = r"^layers\.[0-9]+\.mlp\.experts.[0-9]+\."
    for name, param in model.named_parameters():
        numel = param.numel()
        if regex.match(pat, name):
            expert_tot += numel
        else:
            tot += numel

    assert expert_tot > 0 and expert_tot % num_experts == 0
    return tot + (expert_tot // num_experts * num_active_experts)
