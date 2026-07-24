import os
import time
import numpy as np
from gpt2.model import GPTConfig, GPT
import jax
import tiktoken
# from utils import print_compiling
from flax.training import train_state

# batch_size = 12
batch_size = 2
block_size = 1024
eval_interval = 2000
log_interval = 1
eval_iters = 2 # 200
learning_rate = 6e-4 # max learning rate
dropout=0.0
beta1 = 0.9
beta2 = 0.95
max_new_tokens = 100 # number of tokens generated in each sample
temperature = 0.8 # higher temperature (up to 1) is more random, lower (down to 0) means more greedy
top_k = 200 # retain only the top_k most likely tokens, clamp others to have 0 probability
vocab_size = 50257

print("Initializing a new model from scratch")
model = GPT(GPTConfig(block_size=block_size, dropout=dropout, vocab_size=vocab_size))
state = model.create_state(learning_rate, beta1, beta2)

# dataset = 'shakespeare'
dataset = "random"

if dataset != "random":
    data_dir = os.path.join('data', dataset)
    train_data = np.memmap(os.path.join(data_dir, 'train.bin'), dtype=np.uint16, mode='r')
    val_data = np.memmap(os.path.join(data_dir, 'val.bin'), dtype=np.uint16, mode='r')

def get_batch(split):
    if dataset == "random":
        return tuple(np.random.randint(0, vocab_size, (batch_size, block_size)) for _ in range(2))

    data = train_data if split == 'train' else val_data
    ix = np.random.randint(len(data) - block_size, size=(batch_size,))
    x = np.stack([data[i:i+block_size].astype(np.int32) for i in ix])
    y = np.stack([data[i+1:i+1+block_size].astype(np.int32) for i in ix])
    return x, y


def train_loop():
    val_batch = get_batch('val')
    
    t0 = time.time()
    iter_num = 0
    global state
    while iter_num < 5:
        if iter_num % eval_interval == 0:
            print("evaluating...")
            sample_str = sample(
                state.params,
                jax.random.key(0),
                tokens=val_batch[0][0:1,:5],
            )
            print(f"sample: {sample_str}")
            losses = estimate_loss()
            print(f"step {iter_num}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
    
        loss, state = train_step(state, get_batch('train'))
    
        t1 = time.time()
        dt = t1 - t0
        t0 = t1
        if iter_num % log_interval == 0:
            lossf = loss.item()
            print(f"iter {iter_num}: loss {lossf:.4f}, time {dt*1000:.2f}ms")
        iter_num += 1

tokenizer = tiktoken.get_encoding("gpt2")

def sample(params, key, tokens) -> str:
    tokens = _sample(params, key, tokens)
    return tokenizer.decode(tokens[0])

@jax.jit
# @print_compiling
def _sample(params, key, tokens) -> jax.Array:
    return model.generate(
        key, params, tokens,
        max_new_tokens=max_new_tokens,
        top_k=top_k,
        temperature=temperature
    )

def estimate_loss():
    out = {}
    for split in ['train', 'val']:
        losses = np.zeros(eval_iters)
        for k in range(eval_iters):
            batch = get_batch(split)
            logits, loss = forward(state, batch, train=False)
            losses[k] = float(loss)
        out[split] = losses.mean()
    return out

@jax.jit
# @print_compiling
def train_step(state: train_state.TrainState, batch):
    def loss_fn(params):
        state_ = state.replace(params=params)
        logits, loss = forward(state_, batch, train=True)
        return loss
    grad_fn = jax.value_and_grad(loss_fn)
    loss, grad = grad_fn(state.params)
    state = state.apply_gradients(grads=grad)
    return loss, state

@jax.jit(static_argnames=('train',))
# @print_compiling
def forward(state, batch, *, train: bool):
    inputs, labels = batch
    rngs = {}
    if train and dropout > 0.0:
        rngs['dropout'] = jax.random.fold_in(
            jax.random.key(0), state.step)
    return state.apply_fn(
        {'params': state.params}, 
         inputs, train=train, targets=labels,
         rngs=rngs)

train_loop()
