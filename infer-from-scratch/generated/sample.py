import torch

selector = "sample_gumbel_max_without_softmax"


def sample_multinomial(logits, temperature):
    assert temperature > 0
    probs = torch.softmax(logits / temperature, dim=-1)
    return torch.multinomial(probs, 1).item()


def sample_gumbel_max_without_softmax(logits, temperature):
    assert temperature > 0
    noise = -torch.empty_like(logits).exponential_(1.0).log()
    return (logits / temperature + noise).argmax().item()


def sample(logits, config):
    if config.temperature == 0.0:
        return logits.argmax().item()

    if selector not in globals():
        raise NotImplementedError(f"Unrecognized sampler selector: {selector}")
    return globals()[selector](logits, config.temperature)
