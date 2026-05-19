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


