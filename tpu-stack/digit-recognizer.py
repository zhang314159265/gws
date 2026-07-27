# Accuracy:
#   95.47% accuracy using adamw after 10 epochs
#   93.23% accuracy using sgd after 10 epochs
# Training latency
#   without jax.jit, abount 5.5 seconds per epoch
#   jax.jit train_step, 0.35 seconds per epoch (15.7x speedup)
#   also jax.jit predict, 0.17 seconds per epoch (32.4x speedup)

from tensorflow.keras.datasets import mnist
import jax
import jax.numpy as jnp
import flax.linen as nn
from flax.training import train_state
import optax
import time
import os

profile_dir = "/tmp/jax_profile"
num_profiling_step = 10
do_profile = os.getenv("DO_PROFILE") == "1"

def display_sample(x, y):
    import matplotlib.pyplot as plt
    plt.title(f"Lable {y}")
    plt.imshow(x, cmap="gray")
    plt.show()

def inspect_images(xlist, ylist):
    for i in range(10):
        display_sample(train_x[i], train_y[i])

# print(jax.tree.structure(mnist.load_data()))
(train_x, train_y), (test_x, test_y) = jax.tree.map(
    lambda x: jnp.array(x),
    mnist.load_data()
)
train_x = jnp.array(train_x, dtype=jnp.float32).reshape(train_x.shape[0], -1)
print(f"{train_x.shape=}")
test_x = jnp.asarray(test_x, dtype=jnp.float32).reshape(test_x.shape[0], -1)

# inspect_images(train_x, train_y)

batch_size = 100
nepoch = 10
assert train_x.shape[0] % batch_size == 0
assert test_x.shape[0] % batch_size == 0

def data_loader(selector):
    assert selector in ("train", "test")
    xlist, ylist = (train_x, train_y) if selector == "train" else (test_x, test_y)

    # TODO shuffle?
    idx = 0
    while idx < xlist.shape[0]:
        yield xlist[idx: idx + batch_size], ylist[idx: idx + batch_size]
        idx += batch_size

class Model(nn.Module):
    @nn.compact
    def __call__(self, x):
        x = nn.Dense(100)(x)
        x = nn.relu(x)
        x = nn.Dense(10)(x)
        return x

def compute_loss(params, x, label):
    # forward
    logits = model.apply({"params": params}, x)
    # print(y.shape, y.dtype)

    # loss
    loss = optax.softmax_cross_entropy_with_integer_labels(logits, label).mean()
    return loss

@jax.jit
def predict(state, x, label):
    logits = model.apply({"params": state.params}, x)
    prediction = jnp.argmax(logits, axis=1)
    return (prediction == label).sum()


def compute_accuracy():
    tot = 0
    cor = 0
    for x, label in data_loader("test"):
        tot += x.shape[0]
        cor += predict(state, x, label)

    return cor / tot if tot > 0 else 0

@jax.jit
def train_step(state, x, label):
    loss, grads = jax.value_and_grad(lambda params: compute_loss(params, x, label))(
        state.params,
    )
    state = state.apply_gradients(grads=grads)
    return state, loss


model = Model()
key = jax.random.key(23)
variables = model.init(key, next(data_loader("train"))[0])

state = train_state.TrainState.create(
    apply_fn=model.apply,
    params=variables["params"],
    # tx=optax.adamw(learning_rate=1e-3, b1=0.9, b2=0.95)
    tx=optax.sgd(learning_rate=1e-3)
)

global_step = 0
num_warmup_step = 1
profile_active = False
for epoch_id in range(nepoch):
    start_ts = time.perf_counter()
    for x, label in data_loader("train"):
        # skip warmup (jit compilation)
        if global_step == num_warmup_step and do_profile:
            print(f"Profile will be written to {profile_dir}")
            jax.profiler.start_trace(profile_dir)
            profile_active = True
            
        with jax.profiler.StepTraceAnnotation("train_step", step_num=global_step):
            state, loss = train_step(state, x, label)
        global_step += 1

        if global_step == num_warmup_step + num_profiling_step and profile_active:
            jax.block_until_ready(state)
            jax.profiler.stop_trace()
            profile_active = False
    acc = compute_accuracy()
    elapse = time.perf_counter() - start_ts
    print(f"epoch {epoch_id}: accuracy {acc * 100:.2f}%, elapse {elapse:.3f} seconds")

if profile_active:
    jax.block_until_ready(state)
    jax.profiler.stop_trace()

print("Done")
