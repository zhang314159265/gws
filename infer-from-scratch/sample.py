import torch

def sample(logits, config):
    if config.temperature == 0:
        return logits.argmax().cpu().item()
    # gumbel max:
    # out-token = argmax(pi / qi)
    # = argmax(log(pi) - log(qi))
    # = argmax(xi - log(qi))
    q = torch.empty_like(logits).exponential_()
    return (logits / config.temperature - q.log()).argmax().cpu().item()


