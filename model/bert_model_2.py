"""
Instead of following HF step by step, try to implement Bert myself.
"""

import torch
import math
from torch import nn
torch.set_default_device("cuda")

class config:
    vocab_size = 30522
    hidden_size = 768 # 12 * 64
    intermediate_size = 768 * 4
    num_attention_heads = 12
    pad_token_id = 0
    layer_norm_eps = 1e-12
    dropout_prob = 0.1

    num_hidden_layers = 2 # TODO it was 12
    initializer_range = 0.02
    max_position = 512

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

class Head(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear_1 = nn.Linear(config.hidden_size, config.hidden_size)
        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.linear_2 = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.linear_2.bias = nn.Parameter(torch.zeros(config.vocab_size))

    def forward(self, hidden_state):
        hidden_state = self.linear_1(hidden_state)
        hidden_state = nn.functional.gelu(hidden_state)
        hidden_state = self.layer_norm(hidden_state)
        hidden_state = self.linear_2(hidden_state)
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
        self.head = Head()

        self.apply(_init_weights)

        # tie weights
        self.head.linear_2.weight = self.word_embeddings.weight

    def forward(self, input_ids, labels):
        hidden_state = self.word_embeddings(input_ids)
        hidden_state += self.position_embeddings(self.position_ids[: input_ids.size(1)])
        hidden_state = self.emb_layer_norm(hidden_state)
        hidden_state = self.emb_dropout(hidden_state)

        hidden_state = self.layers(hidden_state)

        pred_score = self.head(hidden_state)
        loss_fn = nn.CrossEntropyLoss()
        loss = loss_fn(pred_score.view(-1, config.vocab_size), labels.view(-1))
        return loss, pred_score

    @staticmethod
    def generate_example(batch_size=2, seq_length=512):
        x = torch.randint(0, config.vocab_size, (batch_size, seq_length))
        label = torch.randint(0, config.vocab_size, (batch_size, seq_length))
        return x, label
        

if __name__ == "__main__":
    from torch._dynamo.testing import reset_rng_state
    from torch._inductor import config as inductor_config
    from torch._inductor import lowering, graph

    lowering.use_two_step_variance = lambda *args, **kwargs: True  # disable welford for now

    # mock GraphLowering.__init__
    old_init = graph.GraphLowering.__init__
    def mock_init(self, gm, *args, **kwargs):
        gm.print_readable()
        return old_init(self, gm, *args, **kwargs)
    graph.GraphLowering.__init__ = mock_init

    torch.set_float32_matmul_precision('high')

    inductor_config.benchmark_kernel = True
    reset_rng_state()
    x, label = Bert.generate_example()
    model = Bert()
    inference = True
    if inference:
        model.eval()
        with torch.no_grad():
            model = torch.compile(model)
            out = model(x, label)
            print(out)
    else:
        model.train()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01, capturable=True, foreach=True)
        for _ in range(3):
            optimizer.zero_grad(True)
            loss, _ = model(x, label)
            loss.backward()
            optimizer.step()
        print(model.word_embeddings.weight)
    print("bye")
