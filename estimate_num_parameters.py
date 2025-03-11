def estimate(V=128256, C=4096, L=32, INTERM_DIM=14336, num_heads=32, num_kv_heads=8, tied_embedding=False):
    tot = 0

    emb = V * C
    tot += emb

    ffw = L * (3 * C * INTERM_DIM)
    tot += ffw

    assert C % num_heads == 0
    head_size = C // num_heads
    tot_head = (num_heads + num_kv_heads * 2) * head_size
    attn = L * (C * tot_head + C * C)
    tot += attn

    per_rms = C
    rms = L * (2 * per_rms) + per_rms
    tot += rms

    if not tied_embedding:
        tot += V * C # the final linear

    return tot

print(f"Number of parameters for llama3.1 8B: {estimate():_}")
print(f"Number of parameters for llama3.1 70B: {estimate(L=80, num_heads=64, num_kv_heads=8, C=8192, INTERM_DIM=28672):_}")
print(f"Number of parameters for llama3.1 405B: {estimate(L=126, num_heads=128, num_kv_heads=8, C=16384, INTERM_DIM=53248):_}")
