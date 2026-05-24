import torch
from trm import Transformer
from rope import Rope
import time

def get_num_params(model):
    tot = 0

    for param in model.parameters():
        tot += param.numel()
    return tot

def _get_model(config):
    state_dict = torch.load(config.checkpoint_file)
    model = Transformer(config)
    model.load_state_dict(state_dict)
    print(f"Num of parameters {get_num_params(model):_}")
    Rope.precompute_cis(config)
    return model

def get_model(config):
    start_ts = time.perf_counter()
    with torch.device("cuda"):
        model = _get_model(config)
    load_time = time.perf_counter() - start_ts
    print(f"Load model takes {load_time:.3f} seconds")
    return model
