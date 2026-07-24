import flax.linen as nn
import jax.numpy as jnp
import jax
import os
import optax

from dataclasses import dataclass
from flax.training import train_state

NUM_EFFECTIVE_LAYER = int(os.getenv("NUM_EFFECTIVE_LAYER", 2))

@dataclass
class GPTConfig:
    block_size: int = 1024  # max sequence length
    vocab_size: int = 50257
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    dropout: float = 0.0
    epsilon: float = 1e-5

class CausalSelfAttention(nn.Module):
    config: GPTConfig

    @nn.compact
    def __call__(self, x, train):
        config = self.config
        n_head = config.n_head

        B, T, C = x.shape
        head_dim = C // n_head
        temperature = jnp.sqrt(head_dim)

        q, k, v = jnp.split(nn.Dense(C * 3)(x), 3, -1)
        q = q.reshape(B, T, n_head, head_dim).swapaxes(1, 2)
        k = k.reshape(B, T, n_head, head_dim).swapaxes(1, 2)
        v = v.reshape(B, T, n_head, head_dim).swapaxes(1, 2)

        s = (q @ k.swapaxes(-1, -2)) / temperature

        p = nn.softmax(jnp.where(
            jnp.tril(jnp.ones((T, T))),
            s,
            float("-inf")
        ))
        p = nn.Dropout(config.dropout, deterministic=not train)(p)

        y = (p @ v).swapaxes(1, 2).reshape(B, T, C)
        y = nn.Dense(C)(y)
        y = nn.Dropout(config.dropout)(y, deterministic=not train)
        return y


class MLP(nn.Module):
    config: GPTConfig

    @nn.compact
    def __call__(self, x, train) -> jax.Array:
        config = self.config
        x = nn.Dense(4 * config.n_embd)(x)
        x = nn.gelu(x)
        x = nn.Dense(config.n_embd)(x)
        x = nn.Dropout(config.dropout, deterministic=not train)(x)
        return x

class Block(nn.Module):
    config: GPTConfig

    @nn.compact
    def __call__(self, x, train):
        config = self.config

        x = x + CausalSelfAttention(config)(nn.LayerNorm(config.epsilon)(x), train=train)
        x = x + MLP(config)(nn.LayerNorm(config.epsilon)(x), train=train)
        return x

class GPT(nn.Module):
    config: GPTConfig

    @nn.compact
    def __call__(self, idx, train, targets=None):
        config = self.config
        B, T = idx.shape
        assert T <= config.block_size

        wte = nn.Embed(config.vocab_size, config.n_embd)
        wpe = nn.Embed(config.block_size, config.n_embd)
        layers = [
            Block(config) for _ in range(config.n_layer)
        ]
        drop = nn.Dropout(config.dropout)
        ln_f = nn.LayerNorm(config.epsilon)

        pos = jnp.arange(0, T)
        tok_emb = wte(idx)
        pos_emb = wpe(pos)

        x = drop(tok_emb + pos_emb, deterministic=not train)
        for layer in layers[:NUM_EFFECTIVE_LAYER]:
            x = layer(x, train=train)

        x = ln_f(x)

        logits = wte.attend(x)

        if targets is not None:
            loss = optax.softmax_cross_entropy_with_integer_labels(
                logits, targets).mean()
        else:
            loss = None

        return logits, loss

    def create_state(self, learning_rate, beta1, beta2):
        variables = self.init(
            jax.random.key(0),
            jnp.ones((1, 1), dtype=jnp.int32),
            train=False
        )
        tx = optax.adamw(learning_rate=learning_rate, b1=beta1, b2=beta2)
        return train_state.TrainState.create(apply_fn=self.apply, params=variables["params"], tx=tx)

    def generate(self, key, params, input_tokens, max_new_tokens, temperature=1.0, top_k=None):
        B, T = input_tokens.shape
        padding = jnp.zeros((B, max_new_tokens), dtype=jnp.int32)
        tokens = jnp.concatenate([input_tokens, padding], axis=-1)
        indexes = jnp.arange(T, T + max_new_tokens)

        # tokens index -> tokens None
        def scan_f(tokens, i):
            step_key = jax.random.fold_in(key, i)

            logits, _ = self.apply({'params': params}, tokens, train=False)
            logits = logits[:, i - 1, :] / temperature

            if top_k is not None:
                top_logits, top_tokens = jax.lax.top_k(logits, min(top_k, logits.shape[-1]))
                token_idx = jax.random.categorical(step_key, top_logits, axis=-1)
                # gather
                next_token = jnp.take_along_axis(top_tokens, token_idx[:, None], axis=-1).squeeze(-1)
            else:
                next_token = jax.random.categorical(step_key, logits, axis=-1)

            tokens = tokens.at[:, i].set(next_token)

            return tokens, None

        tokens, _ = jax.lax.scan(scan_f, tokens, indexes)

        return tokens
