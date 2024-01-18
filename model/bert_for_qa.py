from torch._dynamo.testing import reset_rng_state
import torch
from torch import nn
import math

class config:
    vocab_size = 30522
    num_labels = 2
    num_hidden_layers = 12
    hidden_size = 768 # 12 * 64
    intermediate_size = 768 * 4
    pad_token_id = 0
    layer_norm_eps = 1e-12
    dropout_prob = 0.1
    max_position = 512
    initializer_range = 0.02
    num_attention_heads = 12

class BertLayer(nn.Module):
    """
    A single encoder layer.
    """
    def __init__(self):
        super().__init__()
        self.attention_head_size = config.hidden_size // config.num_attention_heads

        self.query = nn.Linear(config.hidden_size, config.hidden_size)
        self.key = nn.Linear(config.hidden_size, config.hidden_size)
        self.value = nn.Linear(config.hidden_size, config.hidden_size)
        self.score_dropout = nn.Dropout(config.dropout_prob)

        self.attn_linear = nn.Linear(config.hidden_size, config.hidden_size)
        self.attn_layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.attn_dropout = nn.Dropout(config.dropout_prob)

        self.ffn_linear_1 = nn.Linear(config.hidden_size, config.intermediate_size)
        self.ffn_linear_2 = nn.Linear(config.intermediate_size, config.hidden_size)
        self.ffn_layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.ffn_dropout = nn.Dropout(config.dropout_prob)

    def reshape_and_permute(self, x):
        assert x.dim() == 3
        assert x.size(-1) == config.hidden_size
        assert config.hidden_size % config.num_attention_heads == 0
        bs, seqlen, hidden_size = x.shape
        x = x.view(bs, seqlen, config.num_attention_heads, self.attention_head_size)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_state):
        attn_input = hidden_state
        query = self.reshape_and_permute(self.query(hidden_state))
        key = self.reshape_and_permute(self.key(hidden_state))
        value = self.reshape_and_permute(self.value(hidden_state))

        score = torch.matmul(query, key.transpose(-1, -2))

        score = score / math.sqrt(self.attention_head_size)
        score = nn.functional.softmax(score, dim=-1)
        score = self.score_dropout(score)

        hidden_state = torch.matmul(score, value)
        # XXX what's the impact of the contiguous call on inductor
        hidden_state = hidden_state.permute(0, 2, 1, 3).contiguous()
        hidden_state = hidden_state.view(hidden_state.size()[:-2] + (config.hidden_size,))

        # linear/layernorm/dropout
        hidden_state = self.attn_linear(hidden_state)
        hidden_state = self.attn_dropout(hidden_state)
        hidden_state = self.attn_layer_norm(hidden_state + attn_input)

        # ffn
        ffn_input = hidden_state
        hidden_state = self.ffn_linear_1(hidden_state)
        hidden_state = nn.functional.gelu(hidden_state)
        hidden_state = self.ffn_linear_2(hidden_state)
        hidden_state = self.ffn_dropout(hidden_state)
        hidden_state = self.ffn_layer_norm(hidden_state + ffn_input)
        return hidden_state

def _init_weights(module):
    if isinstance(module, nn.Linear):
        module.weight.data.normal_(0, config.initializer_range)
        if module.bias is not None:
            module.bias.data.zero_()
    elif isinstance(module, nn.Embedding):
        module.weight.data.normal_(0, config.initializer_range)
        if module.padding_idx is not None:
            module.weight.data[module.padding_idx].zero_()
    elif isinstance(module, nn.LayerNorm):
        module.bias.data.zero_()
        module.weight.data.fill_(1.0)

class Bert(nn.Module):
    def __init__(self):
        super().__init__()
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)
        self.position_embeddings = nn.Embedding(config.max_position, config.hidden_size)
        self.register_buffer("position_ids", torch.arange(config.max_position), persistent=False)
        self.emb_layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.emb_dropout = nn.Dropout(config.dropout_prob)

        self.layers = nn.Sequential(*(BertLayer() for _ in range(config.num_hidden_layers)))

        self.qa_outputs = nn.Linear(config.hidden_size, config.num_labels)

        self.apply(_init_weights)

    def forward(self, input_ids, start_positions, end_positions):
        hidden_state = self.word_embeddings(input_ids)
        hidden_state += self.position_embeddings(self.position_ids[: input_ids.size(1)])
        hidden_state = self.emb_layer_norm(hidden_state)
        hidden_state = self.emb_dropout(hidden_state)

        hidden_state = self.layers(hidden_state)

        logits = self.qa_outputs(hidden_state)
        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1).contiguous()
        end_logits = end_logits.squeeze(-1).contiguous()

        loss_fn = nn.CrossEntropyLoss()
        total_loss = (loss_fn(start_logits, start_positions) + loss_fn(end_logits, end_positions)) / 2
        return total_loss, start_logits, end_logits

    @staticmethod
    def generate_example(batch_size=2, seq_length=512):
        x = torch.randint(0, config.vocab_size, (batch_size, seq_length))
        start_positions = torch.randint(0, seq_length, (batch_size,))
        end_positions = torch.randint(0, seq_length, (batch_size,))
        return x, start_positions, end_positions

def run_my_model():
    print("call my model")
    torch.set_default_device("cuda")
    reset_rng_state()
    input_ids, start_positions, end_positions = Bert.generate_example()
    reset_rng_state()
    model = Bert().eval()
    reset_rng_state()
    out = model(input_ids=input_ids, start_positions=start_positions, end_positions=end_positions)
    print(f"output:\n{out}")

def run_hf_model(simplify=False):
    # generate same numerical as
    # python benchmarks/dynamo/huggingface.py --backend inductor --float32 --accuracy --only BertForQuestionAnswering --inference
    torch.set_default_device("cuda")
    batch_size = 2
    seq_length = 512
    reset_rng_state()
    input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_length), device="cuda")
    start_positions = torch.randint(0, seq_length, (batch_size,), device="cuda")
    end_positions = torch.randint(0, seq_length, (batch_size,), device="cuda")

    from transformers import BertForQuestionAnswering as model_cls
    config_cls = model_cls.config_class
    reset_rng_state()
    hf_model = model_cls(config_cls()).to(device="cuda").eval()

    reset_rng_state()
    hf_out = hf_model(input_ids=input_ids, start_positions=start_positions, end_positions=end_positions)
    # https://gist.github.com/shunting314/413d291cb4b78bb0ac78cefaacf8c19e
    print(f"hf_out {hf_out}")

    print("bye")

if __name__ == "__main__":
    use_hf_model = False
    if use_hf_model:
        run_hf_model(simplify=True)
    else:
        run_my_model()
