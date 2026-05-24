class config:
    checkpoint_dir = "artifact/Qwen/Qwen3-Coder-30B-A3B-Instruct"
    tokenizer_file = "artifact/Qwen/Qwen3-Coder-30B-A3B-Instruct/tokenizer.json"

    vocab_size = 151936
    num_layers = 48
    num_q_heads = 32
    num_kv_heads = 4
    head_dim = 128
    hidden_size = 2048
    moe_intermediate_size = 768
    num_experts = 128
    num_experts_per_tok = 8
    norm_topk_prob = True
    rms_norm_eps = 1e-6
    rope_theta = 10_000_000
    temperature = 0.7
    max_position_embeddings = 8192

    prompt = "Show me the C code for bubble sort."
