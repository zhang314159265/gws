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


def sample(logits):
    if config.temperature == 0:
        return logits.argmax().cpu().item()
    # gumbel max:
    # out-token = argmax(pi / qi)
    # = argmax(log(pi) - log(qi))
    # = argmax(xi - log(qi))
    q = torch.empty_like(logits).exponential_()
    return (logits / config.temperature - q.log()).argmax().cpu().item()

def generate(prompt):
    all_tokens = tokenizer.encode(prompt)
    print(all_tokens)
    new_tokens = all_tokens
    n_prompt = len(all_tokens)

    prefill_time = None
    decode_start = None
    decode_tokens = 0
    prefill_start = time.perf_counter()

    try:
        while len(all_tokens) < config.max_position_embeddings:
            start_pos = len(all_tokens) - len(new_tokens)
            x = torch.tensor(new_tokens, device="cuda", dtype=torch.int32)
            with torch.no_grad():
                newtoken = sample(model(x, start_pos))
            # .item() inside sample() syncs CUDA, so perf_counter sees real GPU time

            if prefill_time is None:
                prefill_time = time.perf_counter() - prefill_start
                decode_start = time.perf_counter()
            else:
                decode_tokens += 1

            # print(f"new token {newtoken}")
            print(".", end="", flush=True)
            if tokenizer.is_end_token(newtoken):
                print("Encounter end token")
                break
            all_tokens.append(newtoken)
            new_tokens = [newtoken]
    except KeyboardInterrupt:
        print("\n[interrupted -- showing partial output]")

    decode_time = (time.perf_counter() - decode_start) if decode_start else 0.0

    print(tokenizer.decode(all_tokens))
    print()
    print(f"Prompt:  {n_prompt} tokens")
    if prefill_time is not None:
        print(f"Prefill: {prefill_time * 1000:.1f} ms ({n_prompt / prefill_time:.1f} tok/s)")
    if decode_tokens > 0:
        print(f"Decode:  {decode_tokens} tokens in {decode_time:.2f} s ({decode_tokens / decode_time:.1f} tok/s)")

def interactive():
    while True:
        print("> ", end="")
        prompt = input().strip()
        if not prompt:
            print("Done with the interactive mode")
            break
        generate(prompt)

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
        generate(config.prompt)
