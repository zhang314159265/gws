def compute_qkv_proj_size(C, num_heads, num_kv_heads):
    assert C % num_heads == 0
    head_size = C // num_heads
    return (num_heads + num_kv_heads * 2) * head_size

def estimate(V, C, L, INTERM_DIM, num_heads, num_kv_heads, tied_embedding):
    tot = 0

    emb = V * C
    tot += emb

    ffw = L * (3 * C * INTERM_DIM)
    tot += ffw

    qkv_proj_size = compute_qkv_proj_size(C, num_heads, num_kv_heads)
    attn = L * (C * qkv_proj_size + C * C)
    tot += attn

    per_rms = C
    rms = L * (2 * per_rms) + per_rms
    tot += rms

    if not tied_embedding:
        tot += V * C # the final linear

    return tot

llama_3_1_8B = dict(
    V=128256,
    C=4096,
    L=32,
    INTERM_DIM=14336,
    num_heads=32,
    num_kv_heads=8,
    tied_embedding=False,
)

llama_3_1_70B = dict(
    V=128256,
    C=8192,
    L=80,
    INTERM_DIM=28672,
    num_heads=64,
    num_kv_heads=8,
    tied_embedding=False,
)

llama_3_1_405B = dict(
    V=128256,
    C=16384,
    L=126,
    INTERM_DIM=53248,
    num_heads=128,
    num_kv_heads=8,
    tied_embedding=False,
)

def get_model_configs():
    for k, v in dict(globals()).items():
        if len(k) >= 5 and isinstance(v, dict) and 'V' in v and 'C' in v:
            # just do some basic check
            yield k, v

for k, v in get_model_configs():
    print(f"Number of parameters for {k}: {estimate(**v):_}")

def estimate_activation(V, C, L, INTERM_DIM, num_heads, num_kv_heads, tied_embedding):
    """
    This is a rough estimate of number of activations for each token.
    """
    tot = 0

    # ignore input tokens

    # embedding
    tot += C

    # attn rms norm out
    tot += L * C

    # attn qkv projection
    qkv_proj_size = compute_qkv_proj_size(C, num_heads, num_kv_heads)
    tot += L * qkv_proj_size

    # attn output
    tot += L * C

    # attn proj output
    tot += L * C

    # attn skip connection
    tot += L * C

    # ffw rms norm out
    tot += L * C

    # ffw first matmul
    tot += L * INTERM_DIM

    # ffw second matmul
    tot += L * INTERM_DIM

    # ffw third matmul
    tot += L * C

    # ffw skip connection
    tot += L * C

    # final layer norm
    tot += C

    # final projection
    tot += V

    return tot

for k, v in get_model_configs():
    print(f"Number of activations for each token for {k}: {estimate_activation(**v):_}")
