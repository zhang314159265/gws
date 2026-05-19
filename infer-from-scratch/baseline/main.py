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
from args import parse_args
from model import get_model

def interactive():
    while True:
        print("> ", end="")
        torch.manual_seed(1337)
        prompt = input().strip()
        if not prompt:
            print("Done with the interactive mode")
            break
        generate(prompt, tokenizer, model, config)

args = parse_args()
torch.set_default_dtype(torch.bfloat16)
with torch.device("cuda"):
    model = get_model(config)

with torch.device("cuda"):
    if args.interactive:
        print("Interactive mode")
        interactive()
    else:
        generate(config.prompt, tokenizer, model, config)
