from huggingface_hub import snapshot_download
import os
import tiktoken
from tiktoken.load import load_tiktoken_bpe
import torch
from model import Transformer
import torch.nn.functional as F
import time

REPO_ID = "meta-llama/Meta-Llama-3.1-8B-Instruct"
TOKENIZER_FILE = os.path.join("checkpoints", REPO_ID, "tokenizer.model")
CHECKPOINT_PATH = os.path.join("checkpoints", REPO_ID, "model.pth")

torch._inductor.config.coordinate_descent_tuning = True

def multinomial_sample_one_no_sync(probs):
    q = torch.empty_like(probs).exponential_(1)
    return torch.argmax(probs / q, dim=-1, keepdim=True).to(dtype=torch.int)

def logits_to_probs(logits, temperature, top_k):
    logits = logits / max(temperature, 1e-5)

    if top_k is not None:
        v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
        pivot = v.select(-1, -1).unsqueeze(-1)
        logits = torch.where(logits < pivot, -float("inf"), logits)
    probs = F.softmax(logits, dim=-1)
    return probs

def sample(logits, temperature, top_k):
    probs = logits_to_probs(logits[:, -1], temperature, top_k)
    idx_next = multinomial_sample_one_no_sync(probs)
    return idx_next, probs

def _load_model(checkpoint_path, device, precision):
    with torch.device("meta"):
        model = Transformer.create()

    checkpoint = torch.load(checkpoint_path, mmap=True, weights_only=True)
    model.load_state_dict(checkpoint, assign=True)
    model = model.to(device=device, dtype=precision)
    return model.eval()

class Tokenizer:
    # copied literally from gpt-fast
    pat_str = r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"  # noqa: E501

    def __init__(self, model_path):
        self.model_path = model_path
        mergeable_ranks = load_tiktoken_bpe(model_path)
        num_base_tokens = len(mergeable_ranks)

        special_tokens = [
        ]
        self.special_tokens = {
            token: num_base_tokens + i for i, token in enumerate(special_tokens)
        }
        self.model = tiktoken.Encoding(
            name="",
            pat_str=self.pat_str,
            mergeable_ranks=mergeable_ranks,
            special_tokens=self.special_tokens,
        )

    def encode(self, text):
        return self.model.encode(text)

    def decode(self, tokens):
        return self.model.decode(tokens) 

def prefill(model, x, input_pos, **sampling_kwargs):
    logits = model(x, input_pos)
    return sample(logits, **sampling_kwargs)[0]

def decode_one_token(model, x, input_pos, **sampling_kwargs):
    assert input_pos.shape[-1] == 1

    logits = model(x, input_pos)
    return sample(logits, **sampling_kwargs)

def decode_n_tokens(model, cur_token, input_pos, num_new_tokens, **sampling_kwargs):
    new_tokens, new_probs = [], []

    for i in range(num_new_tokens):
        next_token, next_prob = decode_one_token(
            model, cur_token, input_pos, **sampling_kwargs
        )
        input_pos += 1
        new_tokens.append(next_token.clone())
        new_probs.append(next_prob.clone())
        cur_token = next_token.clone()

    return new_tokens, new_probs

@torch.no_grad()
def generate(
    model,
    prompt,
    max_new_tokens=200,
    batch_size=1,
    **sampling_kwargs
):
    T = prompt.size(-1)
    T_new = T + max_new_tokens
    max_seq_length = min(T_new, model.config.block_size)
    device = prompt.device

    with torch.device(device):
        model.setup_caches(max_batch_size=batch_size, max_seq_length=max_seq_length)
    seq = torch.empty(batch_size, T_new, dtype=prompt.dtype, device=device)
    prompt = prompt.view(1, -1).repeat(batch_size, 1)
    seq[:, :T] = prompt
    input_pos = torch.arange(0, T, device=device)

    # prefill
    next_token = prefill(model, prompt, input_pos, **sampling_kwargs).clone()
    seq[:, T] = next_token.squeeze()

    # decode
    input_pos = torch.tensor([T], device=device, dtype=torch.int)
    generated_tokens, _ = decode_n_tokens(model, next_token.view(batch_size, -1), input_pos, max_new_tokens - 1, **sampling_kwargs)
    seq[:, T + 1:] = torch.cat(generated_tokens, dim=-1)

    return seq

def download(repo_id=REPO_ID):
    local_dir = f"checkpoints/{repo_id}"
    if os.path.exists(local_dir):
        print("Checkpoint is already cached")
    else:
        snapshot_download(repo_id, local_dir=local_dir)

def setup_to_use_kernel_lib():
    from model import RMSNorm
    RMSNorm.forward = RMSNorm.forward_kernel

def main(prompt, num_samples, compile, compile_prefill, use_kernel_lib):
    download()

    if use_kernel_lib:
        setup_to_use_kernel_lib()
    
    tokenizer = Tokenizer(TOKENIZER_FILE)
    device = "cuda"
    encoded = torch.tensor(tokenizer.encode(prompt), dtype=torch.int, device=device)
    print(f"{encoded=}")
    prompt_length = encoded.size(-1)
    precision = torch.bfloat16
    model = _load_model(CHECKPOINT_PATH, device, precision)
    torch.manual_seed(1234)

    if compile:
        global decode_one_token, prefill
        decode_one_token = torch.compile(decode_one_token, mode="reduce-overhead", fullgraph=True)

        if compile_prefill:
            prefill = torch.compile(prefill, fullgraph=True, dynamic=True)

    start = -1 if compile else 0

    for i in range(start, num_samples):
        print(f"-------- iteration {i} ----------")
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        y = generate(model, encoded, temperature=0.8, top_k=200)
        torch.cuda.synchronize()

        if i == -1:
            print(f"Compilation time: {time.perf_counter() - t0:.2f} seconds")
            continue
        t = time.perf_counter() - t0

        tokens_generated = y.size(-1) - prompt_length
        generated_tokens_sec = tokens_generated / t 
        print("+ ANSWER:")
        print(tokenizer.decode(y[0].tolist()))
        print("+ STATS:")
        print(f"Time for inference {i + 1}: {t:.02f} sec total, {generated_tokens_sec:.02f} tokens/sec")
        print()
    print("bye")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="GPT Inference.")
    parser.add_argument("--prompt", type=str, default="How to estimate pi?", help="Input prompt.")
    parser.add_argument("--num_samples", type=int, default=1, help="Number of smaples.")
    parser.add_argument("--compile", action="store_true", help="Whether to compile the model.")
    parser.add_argument("--compile_prefill", action="store_true", help="Whether to compile the prefill.")
    parser.add_argument("--use-kernel-lib", action="store_true", help="Whether to use the kernel library")
    args = parser.parse_args()
    main(args.prompt, args.num_samples, args.compile, args.compile_prefill, args.use_kernel_lib)
