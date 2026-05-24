import torch
import time
from sample import sample
import contextlib

@torch.no_grad
def generate(args, prompt, tokenizer, model, config):
    torch.manual_seed(1337)
    tokens = tokenizer.encode(prompt)
    print(tokens)
    n_input_tokens = len(tokens)

    torch.cuda.synchronize()
    prefill_start = time.perf_counter()
    tokens.append(sample(model(torch.tensor(tokens, dtype=torch.int32), 0), config))
    torch.cuda.synchronize()  # optional since .item in sample causes sync
    prefill_time = time.perf_counter() - prefill_start
    decode_start = time.perf_counter()

    cap = len(tokens) + 15 if args.profile else config.max_position_embeddings
    profile_ctx = torch.profiler.profile() if args.profile else contextlib.nullcontext()
    try:
        with profile_ctx:
            while len(tokens) <= cap:
                print(".", end="", flush=True)

                record_ctx = torch.profiler.record_function(f"tok_{len(tokens)}") if args.profile else contextlib.nullcontext()
                with record_ctx:
                    new_token = sample(model(torch.tensor(tokens[-1:], dtype=torch.int32), start_pos=len(tokens) - 1), config)
                if tokenizer.is_end_token(new_token):
                    break
                tokens.append(new_token)
    except KeyboardInterrupt:
        print("\n[Interrupted. Show partial output]")

    decode_time = time.perf_counter() - decode_start

    print()
    print(tokenizer.decode(tokens))

    n_output_tokens = len(tokens) - n_input_tokens

    print(f"Prefill {n_input_tokens} tokens in {prefill_time * 1000:.1f} ms ({n_input_tokens / prefill_time:.1f} tok/s)")
    print(f"Decode {n_output_tokens} tokens in {decode_time * 1000:.1f} ms ({n_output_tokens / decode_time:.1f} tok/s)")
    if args.profile:
        path = "/tmp/trace.json"
        profile_ctx.export_chrome_trace(path)
        print(f"Profile trace written to {path}")

def interactive(args, tokenizer, model, config):
    while True:
        print("> ", end="")
        prompt = input()
        if prompt is None:
            break
        generate(args, prompt, tokenizer, model, config)
