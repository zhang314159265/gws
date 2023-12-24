# - match numerical with pt2 huggingface.py for inference accuracy test.
# TODO support training
# TODO do perf test (with larger batch size)

import sys
sys.path.append(".")

import torch
from transformers import BertForMaskedLM as model_cls
from model.bert_model import MyBertModel
from torch._dynamo.testing import reset_rng_state
from torch._dynamo.utils import same

def download_model(model_cls, config):
    return model_cls(config)

model_name = "BertForMaskedLM"
config = model_cls.config_class()
reset_rng_state()
model = download_model(model_cls, config)
device = "cuda"
dtype = torch.float32
model = model.to(device, dtype=dtype)
# batch_size = 16 # batch size is 1 for accuracy of HF
batch_size = 1
seq_length = 512
vocab_size = model.config.vocab_size
reset_rng_state()
input = torch.randint(0, vocab_size, (batch_size, seq_length), device=device, dtype=torch.int64, requires_grad=False)
labels = torch.randint(0, vocab_size, (batch_size, seq_length), device=device, dtype=torch.int64, requires_grad=False)
input_dict = {
    "input_ids": input,
    "labels": labels,
}

# TODO: should this be done before constructing the model object?
for attr in dir(config):
    if "drop" in attr and isinstance(getattr(config, attr), float):
        setattr(config, attr, 1e-30)

model.eval()
ref_output = model(**input_dict)

reset_rng_state()
my_model = MyBertModel().to(device="cuda").eval()
act_output = my_model(**input_dict)

if not same(ref_output, act_output):
    breakpoint()

assert same(ref_output, act_output)

print("bye")
