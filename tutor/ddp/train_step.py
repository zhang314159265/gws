import torch.nn.functional as F

from .utils import print0, printall
from ..fsdp.util import compute_tensor_hash

def train_step(model, datagen, optim, *, inject_all_reduce=None):
    # train for two steps
    x, label = datagen.generate(32)
    printall(f"hash of x {compute_tensor_hash(x)} label {compute_tensor_hash(label)}")
    probs = model(x)
    loss = F.binary_cross_entropy(probs.flatten(), label)
    loss.backward()
    if inject_all_reduce:
        inject_all_reduce()
    optim.step()
    optim.zero_grad(set_to_none=True)
