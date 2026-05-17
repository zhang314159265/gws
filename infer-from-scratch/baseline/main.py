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

class config:
    checkpoint_file = "artifact/meta-llama/Meta-Llama-3-8B-Instruct/original/consolidated.00.pth"
    tokenizer_file = "artifact/meta-llama/Meta-Llama-3-8B-Instruct/original/tokenizer.model"
    vocab_size = 128256
    num_layers = 32
    num_q_heads = 32
    num_kv_heads = 8
    rms_norm_eps = 1e-5
    hidden_size = 4096
    intermediate_size = int(4096 * 3.5)
    rope_theta = 500_000
    temperature = 0.7
    max_position_embeddings = 8192

    # prompt = "Show me how quick-sort works."
    # prompt = "What's the value of pi in mathematics?"
    # prompt = "Can you explain FFT to me?"
    # prompt = "Can you explain S&P index to me?"
    # prompt = "Translate 'hello' to Chinese."
    prompt = "Show me the C code for bubble sort."
    # prompt = "Explain KL-divergence."

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
