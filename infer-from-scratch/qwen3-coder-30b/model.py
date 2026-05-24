import json
import os
import time
import torch
from safetensors import safe_open
from trm import Transformer
from rope import Rope
from num_params import get_num_params, get_num_active_params

def _get_model(config):
    model = Transformer(config)
    print(f"Num params {get_num_params(model):_}")
    print(f"Num active params {get_num_active_params(model, config):_}")
    state_dict = load_safetensors(config)
    state_dict = remap_state_dict(state_dict)
    model.load_state_dict(state_dict)

    Rope.precompute_cis(config)

    for sub in model.modules():
        if hasattr(sub, "_group_expert_weights"):
            sub._group_expert_weights()
    return model

def get_model(config):
    start_ts = time.perf_counter()
    with torch.device("cuda"):
        model = _get_model(config)
    load_time = time.perf_counter() - start_ts
    print(f"Load model takes {load_time:.3f} seconds")
    return model

def remap_state_dict(state_dict):
    new_st = {}
    prefix = "model."
    for key, val in state_dict.items():
        if key.startswith(prefix):
            key = key[len(prefix):]
        new_st[key] = val
    return new_st

def load_safetensors(config):
    with open(os.path.join(config.checkpoint_dir, "model.safetensors.index.json")) as f:
        weight_map = json.load(f)["weight_map"]
    shards = set(weight_map.values())
    state_dict = {}
    for shard in shards:
        with safe_open(os.path.join(config.checkpoint_dir, shard), framework="pt") as f:
            for tensor_name in f.keys():
                state_dict[tensor_name] = f.get_tensor(tensor_name)
    return state_dict
