import torch
from torch import nn
import functools
import math
import argparse
import time
from config import config
from tokenizer import Tokenizer
tokenizer = Tokenizer(config.tokenizer_file)

from attn import Attention
from ffn import FeedForward
from rope import Rope
from trm_layer import TransformerLayer
from trm import Transformer
from generate import generate, interactive
from args import parse_args
from model import get_model

args = parse_args()
torch.set_default_dtype(torch.bfloat16)
with torch.device("cuda"):
    model = get_model(config)

with torch.device("cuda"):
    if args.interactive:
        interactive(tokenizer, model, config)
    else:
        generate(config.prompt, tokenizer, model, config)
