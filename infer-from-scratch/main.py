import torch
from torch import nn
import functools
import math

torch.manual_seed(1337)

@functools.cache
def print_once(msg):
    print(msg)

class config:
    checkpoint_file = "artifact/meta-llama/Meta-Llama-3-8B/original/consolidated.00.pth"
    tokenizer_file = "artifact/meta-llama/Meta-Llama-3-8B/original/tokenizer.model"
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
    # prompt = "Show me the C code for bubble sort."
    prompt = "Explain KL-divergence."

class Tokenizer:
    def __init__(self):
        import tiktoken
        from tiktoken.load import load_tiktoken_bpe

        path = config.tokenizer_file
        # copied from the llama3 repo
        pat_str = r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"
        mergeable_ranks = load_tiktoken_bpe(path)

        num_special_tokens = 256
        special_tokens = [
            "<|begin_of_text|>",
            "<|end_of_text|>",
            "<|reserved_special_token_0|>",
            "<|reserved_special_token_1|>",
            "<|reserved_special_token_2|>",
            "<|reserved_special_token_3|>",
            "<|start_header_id|>",
            "<|end_header_id|>",
            "<|reserved_special_token_4|>",
            "<|eot_id|>",  # end of turn
        ] + [
            f"<|reserved_special_token_{i}|>"
            for i in range(5, num_special_tokens - 5)
        ]

        assert len(mergeable_ranks) == 128000
        special_tokens_dict = {
            s: i + 128000 for i, s in enumerate(special_tokens)
        }

        self.model = tiktoken.Encoding(
            name="mytokenizer",
            pat_str=pat_str,
            mergeable_ranks=mergeable_ranks,
            special_tokens=special_tokens_dict,
        )
        print(f"vocab size is {self.model.n_vocab}")

        self.bos = 128000
        self.eos = 128001
        self.eot = 128009

    def encode(self, s, bos=True):
        return [self.bos] + self.model.encode(s)

    def decode(self, tokens):
        return self.model.decode(tokens)

    def is_end_token(self, tok):
        return tok == self.eos or tok == self.eot
        
tokenizer = Tokenizer()

# test the round-trip of tokenizer
# print(tokenizer.decode(tokenizer.encode(config.prompt)))

class FeedForward(nn.Module):
    def __init__(self):
        super().__init__()
        # gate
        self.w1 = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        # up
        self.w3 = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        # down
        self.w2 = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)

    def forward(self, x):
        gate = self.w1(x)
        up = self.w3(x)
        return self.w2(gate * torch.sigmoid(gate) * up)

class Rope:
    freqs_cis = None

    @classmethod
    def precompute_freqs_cis(cls, max_seq_len=config.max_position_embeddings):
        head_dim = config.hidden_size // config.num_q_heads
        freqs = 1.0 / config.rope_theta ** (torch.arange(0, head_dim, 2).float() / head_dim)
        t = torch.arange(max_seq_len).float()
        angles = torch.outer(t, freqs)
        cls.freqs_cis = torch.polar(torch.ones_like(angles), angles)

    @classmethod
    def apply_rope(cls, q, start_pos):
        assert cls.freqs_cis is not None
        seqlen, num_head, head_dim = q.shape
        freqs_cis = cls.freqs_cis[start_pos : start_pos + seqlen]
        q = q.transpose(0, 1)  # num_head, seqlen, head_dim
        assert seqlen <= freqs_cis.size(0)

        q = q.contiguous().view(num_head, seqlen, -1, 2)
        c_q = torch.view_as_complex(q.float())
        q = torch.view_as_real(c_q * freqs_cis).flatten(-2).to(dtype=q.dtype)
        q = q.transpose(0, 1)
        return q

class Attention(nn.Module):
    def __init__(self):
        super().__init__()
        self.head_dim = config.hidden_size // config.num_q_heads
        self.wq = nn.Linear(config.hidden_size, self.head_dim * config.num_q_heads, bias=False)
        self.wk = nn.Linear(config.hidden_size, self.head_dim * config.num_kv_heads, bias=False)
        self.wv = nn.Linear(config.hidden_size, self.head_dim * config.num_kv_heads, bias=False)
        self.wo = nn.Linear(config.hidden_size, config.hidden_size, bias=False)

        self.kvcache = torch.empty(2, config.max_position_embeddings, config.num_kv_heads, self.head_dim)

    def forward(self, x, start_pos):
        q, k, v = self.wq(x), self.wk(x), self.wv(x)
        g = config.num_q_heads // config.num_kv_heads

        seqlen_q = x.size(0)
        seqlen_kv = seqlen_q + start_pos

        # a naive attention for now
        q = q.view(seqlen_q, -1, self.head_dim)
        k = k.view(seqlen_q, -1, self.head_dim)
        v = v.view(seqlen_q, -1, self.head_dim)

        q = Rope.apply_rope(q, start_pos)
        k = Rope.apply_rope(k, start_pos)

        # handle kv cache
        self.kvcache[0, start_pos: start_pos + seqlen_q, :, :] = k
        self.kvcache[1, start_pos: start_pos + seqlen_q, :, :] = v
        k = self.kvcache[0, : start_pos + seqlen_q, :, :]
        v = self.kvcache[1, : start_pos + seqlen_q, :, :]

        # expand k and v
        def _expand_kv(kv):
            kv = kv.view(seqlen_kv, -1, 1, self.head_dim)
            kv = kv.expand(-1, -1, g, -1).contiguous()
            kv = kv.view(seqlen_kv, -1, self.head_dim)
            return kv

        k = _expand_kv(k)
        v = _expand_kv(v)

        q = q.transpose(0, 1)  # num_head, seqlen, head_dim
        k = k.transpose(0, 1)
        v = v.transpose(0, 1)

        p = q @ k.transpose(1, 2)  # bmm
        p /= math.sqrt(self.head_dim)

        # apply causal mask
        mask = torch.full([seqlen_kv, seqlen_kv], float("-inf"), device="cuda")
        mask = torch.triu(mask, diagonal=1)[-seqlen_q:]
        s = torch.softmax(p + mask[None, :, :], dim=-1)

        # compute output
        o = s @ v
        o = o.transpose(0, 1).contiguous().view(seqlen_q, -1)
        o = self.wo(o)
        return o

class TransformerLayer(nn.Module):
    def __init__(self):
        super().__init__()
      
        self.attention_norm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.attention = Attention()
        self.ffn_norm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.feed_forward = FeedForward()

    def forward(self, x, start_pos):
        h = x + self.attention(self.attention_norm(x), start_pos)
        y = h + self.feed_forward(self.ffn_norm(h))
        return y


class Model(nn.Module):
    def __init__(self):
        super().__init__()

        # embed
        self.tok_embeddings = nn.Embedding(config.vocab_size, config.hidden_size)
        # layers
        self.layers = nn.ModuleList([
            TransformerLayer()
            for _ in range(config.num_layers)
        ])

        # unembed
        self.norm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.output = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    def forward(self, prompt, start_pos):
        x = self.tok_embeddings(prompt)
        for layer in self.layers:
            x = layer(x, start_pos)
        x = self.norm(x)
        x = self.output(x[-1])  # only compute logits for the last token
        return x

def sample(logits):
    if config.temperature == 0:
        return logits.argmax().cpu().item()
    # gumbel max:
    # out-token = argmax(pi / qi)
    # = argmax(log(pi) - log(qi))
    # = argmax(xi - log(qi))
    q = torch.empty_like(logits).exponential_()
    return (logits / config.temperature - q.log()).argmax().cpu().item()

state_dict = torch.load(config.checkpoint_file)
torch.set_default_dtype(torch.bfloat16)
with torch.device("cuda"):
    model = Model()
    Rope.precompute_freqs_cis()
model.load_state_dict(state_dict)
all_tokens = tokenizer.encode(config.prompt)
print(all_tokens)
new_tokens = all_tokens
while len(all_tokens) < config.max_position_embeddings:
    start_pos = len(all_tokens) - len(new_tokens)
    x = torch.tensor(new_tokens, device="cuda", dtype=torch.int32)
    with torch.no_grad():
        newtoken = sample(model(x, start_pos))
    # print(f"new token {newtoken}")
    print(".", end="", flush=True)
    if tokenizer.is_end_token(newtoken):
        print("Encounter end token")
        break
    all_tokens.append(newtoken)
    new_tokens = [newtoken]

print(tokenizer.decode(all_tokens))
