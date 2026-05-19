import torch
from torch import nn
import functools
import math
import argparse
import time

torch.manual_seed(1337)

@functools.cache
def print_once(msg):
    print(msg)

from config import config
from tokenizer import Tokenizer
tokenizer = Tokenizer(config.tokenizer_file)

from attn import Attention
from ffn import FeedForward
from rope import Rope
from trm_layer import TransformerLayer
from trm import Transformer
from generate import generate


def interactive():
    while True:
        print("> ", end="")
        torch.manual_seed(1337)
        prompt = input().strip()
        if not prompt:
            print("Done with the interactive mode")
            break
        generate(prompt, tokenizer, model, config)

def parse_args():
    parser = argparse.ArgumentParser(description="Inference tool.")
    parser.add_argument("--interactive", action="store_true", help="Whether to run the tool in interactive mode")
    return parser.parse_args()

args = parse_args()

state_dict = torch.load(config.checkpoint_file)
torch.set_default_dtype(torch.bfloat16)
with torch.device("cuda"):
    model = Transformer(config)
    Rope.precompute_cis(config)
model.load_state_dict(state_dict)

with torch.device("cuda"):
    if args.interactive:
        print("Interactive mode")
        interactive()
    else:
        generate(config.prompt, tokenizer, model, config)
