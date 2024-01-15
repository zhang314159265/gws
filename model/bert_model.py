"""
Implemented by following HF step by step.
"""

import torch
from torch import nn
import math

class config:
    hidden_size = 768 
    intermediate_size = 3072
    hidden_act = 'gelu'
    num_attention_heads = 12
    vocab_size = 30522
    type_vocab_size = 2
    layer_norm_eps = 1e-12
    pad_token_id = 0
    max_position_embeddings = 512
    position_embedding_type = "absolute"
    initializer_range = 0.02
    num_hidden_layers = 12

    hidden_dropout_prob = 0.1 # TODO may need tune dropout prob
    attention_probs_dropout_prob = 0.1

    # Simplify the model to ease testing. Will cause numerical change
    # compared to the original Bert implementation.
    simplify = False

def _init_weights(module):
    if getattr(module, "_inititialized_for_bert", False):
        return
    if isinstance(module, nn.Linear):
        module.weight.data.normal_(mean=0.0, std=config.initializer_range)
        if module.bias is not None:
            module.bias.data.zero_()
    elif isinstance(module, nn.Embedding):
        module.weight.data.normal_(mean=0.0, std=config.initializer_range)
        if module.padding_idx is not None:
            module.weight.data[module.padding_idx].zero_()
    elif isinstance(module, nn.LayerNorm):
        module.bias.data.zero_()
        module.weight.data.fill_(1.0)

    module._inititialized_for_bert = True

class BertSelfAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.query = nn.Linear(config.hidden_size, config.hidden_size)
        self.key = nn.Linear(config.hidden_size, config.hidden_size)
        self.value = nn.Linear(config.hidden_size, config.hidden_size)

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)
        
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = config.hidden_size // self.num_attention_heads

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states):
        mixed_query_layer = self.query(hidden_states)
        key_layer = self.transpose_for_scores(self.key(hidden_states))
        value_layer = self.transpose_for_scores(self.value(hidden_states))
        query_layer = self.transpose_for_scores(mixed_query_layer)
        attention_scores = torch.matmul(query_layer, key_layer.transpose(2, 3))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        attention_probs = nn.functional.softmax(attention_scores, dim=-1)
        attention_probs = self.dropout(attention_probs)
        context_layer = torch.matmul(attention_probs, value_layer)
        # XXX what's the impact of contiguous call for inductor
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        context_layer = context_layer.view(context_layer.size()[:-2] + (config.hidden_size,))
        outputs = context_layer,

        return outputs

class BertSelfOutput(nn.Module):
    def __init__(self):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states

class BertAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.self = BertSelfAttention()
        self.output = BertSelfOutput()

    def forward(self, hidden_states):
        self_outputs = self.self(
            hidden_states,
        )
        attention_output = self.output(self_outputs[0], hidden_states)
        outputs = (attention_output,) + self_outputs[1:]
        return outputs

# class GELUActivation(nn.Module):
#     def __init__(self):
#         super().__init__()

class BertIntermediate(nn.Module):
    def __init__(self):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.intermediate_size)
        # self.act = nn.GELUActivation()
        self.act = nn.functional.gelu

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.act(hidden_states)
        return hidden_states

class BertOutput(nn.Module):
    def __init__(self):
        super().__init__()
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states

class BertLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.attention = BertAttention()
        self.intermediate = BertIntermediate()
        self.output = BertOutput()

    def forward(self,
        hidden_states,
    ):
        self_attention_outputs = self.attention(hidden_states)
        attention_output = self_attention_outputs[0]
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        return layer_output,

class BertEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer = nn.ModuleList([BertLayer() for _ in range(config.num_hidden_layers)])

    def forward(self, hidden_states):
        for i, layer_module in enumerate(self.layer):
            layer_outputs = layer_module(
                hidden_states,
            )
            hidden_states = layer_outputs[0]
        return hidden_states

class BertEmbeddings(nn.Module):
    def __init__(self):
        super().__init__()
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.register_buffer("position_ids", torch.arange(config.max_position_embeddings).expand((1, -1)), persistent=False)

        if not config.simplify:
            self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size)
            self.register_buffer(
                "token_type_ids", torch.zeros(self.position_ids.size(), dtype=torch.long), persistent=False)

        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, input_ids, token_type_ids=None, position_ids=None, inputs_embeds=None, past_key_values_length = 0):
        assert inputs_embeds is None
        assert position_ids is None
        assert past_key_values_length == 0

        input_shape = input_ids.size()
        seq_length = input_shape[1]
        inputs_embeds = self.word_embeddings(input_ids)
        position_ids = self.position_ids[:, past_key_values_length : seq_length + past_key_values_length]
        position_embeddings = self.position_embeddings(position_ids)

        embeddings = inputs_embeds

        if not config.simplify:
            assert token_type_ids is not None

            token_type_embeddings = self.token_type_embeddings(token_type_ids)

            embeddings += token_type_embeddings

        embeddings += position_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings

class BertModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.embeddings = BertEmbeddings()
        self.encoder = BertEncoder()

        if not config.simplify:
            self.apply(_init_weights)

    def forward(self, input_ids, token_type_ids=None):
        input_shape = input_ids.size()
        batch_size, seq_length = input_shape
        if token_type_ids is None and not config.simplify:
            assert hasattr(self.embeddings, "token_type_ids")
            buffered_token_type_ids = self.embeddings.token_type_ids[:, :seq_length]
            buffered_token_type_ids_expanded = buffered_token_type_ids.expand(batch_size, seq_length)
            token_type_ids = buffered_token_type_ids_expanded

        embedding_output = self.embeddings(
            input_ids=input_ids,
            token_type_ids=token_type_ids,
        )
        encoder_outputs = self.encoder(
            embedding_output
        )
        return encoder_outputs

class BertPredictionHeadTransform(nn.Module):
    def __init__(self):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.transform_act_fn = nn.functional.gelu
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.transform_act_fn(hidden_states)
        hidden_states = self.LayerNorm(hidden_states)
        return hidden_states

class BertLMPredictionHead(nn.Module):
    def __init__(self):
        super().__init__();
        self.transform = BertPredictionHeadTransform()
        self.decoder = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.bias = nn.Parameter(torch.zeros(config.vocab_size))
        self.decoder.bias = self.bias

    def forward(self, hidden_states):
        hidden_states = self.transform(hidden_states)
        hidden_states = self.decoder(hidden_states)
        return hidden_states

BertOnlyMLMHead = BertLMPredictionHead

class BertForMaskedLM(nn.Module):
    def __init__(self):
        super().__init__()
        self.bert = BertModel()
        self.cls = BertOnlyMLMHead()
        self.apply(_init_weights) # to match HF
        # tie weights
        self.cls.decoder.weight = self.bert.embeddings.word_embeddings.weight

    def forward(self, input_ids, labels):
        outputs = self.bert(input_ids)
        sequence_output = outputs
        prediction_scores = self.cls(sequence_output)

        assert labels is not None
        loss_fct = nn.CrossEntropyLoss()
        masked_lm_loss = loss_fct(prediction_scores.view(-1, config.vocab_size), labels.view(-1))
        return {
            "loss": masked_lm_loss,
            "logits": prediction_scores,
        }

MyBertModel = BertForMaskedLM

if __name__ == "__main__":
    from torch._dynamo.testing import reset_rng_state
    torch.set_float32_matmul_precision('high')
    reset_rng_state()
    config.simplify = True
    batch_size = 2
    seq_length = 512
    device = "cuda"
    input = torch.randint(0, config.vocab_size, (batch_size, seq_length), device=device, dtype=torch.int64, requires_grad=False)
    labels = torch.randint(0, config.vocab_size, (batch_size, seq_length), device=device, dtype=torch.int64, requires_grad=False)
    input_dict = {
        "input_ids": input,
        "labels": labels,
    }
    torch.set_default_device("cuda")
    model = MyBertModel().to("cuda")

    inference = True
    if inference:
        model.eval()
        out = model(**input_dict)
        print(out)
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01, capturable=True, foreach=True)
        for _ in range(3):
            optimizer.zero_grad(True)
            pred = model(**input_dict)
            loss = pred[next(iter(pred))]
            loss.backward()
            optimizer.step()
        print(model.bert.embeddings.word_embeddings.weight)
    print("bye")
