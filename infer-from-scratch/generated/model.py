import torch
import json
import os
import time
from safetensors import safe_open
from trm import Transformer
from rope import Rope


def load_safetensors(checkpoint_dir):
    index_path = os.path.join(checkpoint_dir, "model.safetensors.index.json")
    with open(index_path) as f:
        index = json.load(f)

    state_dict = {}
    loaded_files = set()
    for weight_name, filename in index["weight_map"].items():
        if filename not in loaded_files:
            filepath = os.path.join(checkpoint_dir, filename)
            with safe_open(filepath, framework="pt") as f:
                for key in f.keys():
                    state_dict[key] = f.get_tensor(key)
            loaded_files.add(filename)
    return state_dict


def remap_state_dict(state_dict):
    new_sd = {}
    for k, v in state_dict.items():
        new_key = k
        if new_key.startswith("model."):
            new_key = new_key[len("model."):]
        new_sd[new_key] = v
    return new_sd


def _get_model(config):
    state_dict = load_safetensors(config.checkpoint_dir)
    state_dict = remap_state_dict(state_dict)
    model = Transformer(config)
    model.load_state_dict(state_dict)
    Rope.precompute_cis(config)
    return model


def get_model(config):
    start_ts = time.perf_counter()
    with torch.device("cuda"):
        model = _get_model(config)
    load_time = time.perf_counter() - start_ts
    print(f"Load model takes {load_time:.3f} seconds")
    return model
