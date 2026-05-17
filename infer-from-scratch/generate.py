import torch
import time
from sample import sample

def generate(prompt, tokenizer, model, config):
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
                newtoken = sample(model(x, start_pos), config)
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


