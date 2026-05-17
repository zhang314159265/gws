import torch
import time
from sample import sample

@torch.no_grad
def generate(prompt, tokenizer, model, config):
    tokens = tokenizer.encode(prompt)
    print(tokens)
    n_input_tokens = len(tokens)

    torch.cuda.synchronize()
    prefill_start = time.perf_counter()
    tokens.append(sample(model(torch.tensor(tokens, dtype=torch.int32), 0), config))
    torch.cuda.synchronize()  # optional since .item in sample causes sync
    prefill_time = time.perf_counter() - prefill_start
    decode_start = time.perf_counter()

    try:
        while len(tokens) <= config.max_position_embeddings:
            print(".", end="", flush=True)
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
