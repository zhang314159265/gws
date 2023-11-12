# TODO: repro pt2 bench script first

import torch
from transformers import BertForMaskedLM as model_cls
from torch._dynamo.testing import reset_rng_state

def download_model(model_cls, config):
    return model_cls(config)

model_name = "BertForMaskedLM"
config = model_cls.config_class()
model = download_model(model_cls, config)
device = "cuda"
dtype = torch.float32
model = model.to(device, dtype=dtype)
batch_size = 16
seq_length = 512
vocab_size = model.config.vocab_size
reset_rng_state()
input = torch.randint(0, vocab_size, (batch_size, seq_length), device=device, dtype=torch.int64, requires_grad=False)
labels = torch.randint(0, vocab_size, (batch_size, seq_length), device=device, dtype=torch.int64, requires_grad=False)
input_dict = {
    "input_ids": input,
    "labels": labels,
}

# TODO: override dropout to 1e-30

breakpoint()

print("bye")
