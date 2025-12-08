import torch.nn.functional as F

def train_step(model, datagen, optim, *, inject_all_reduce=None):
    # train for two steps
    x, label = datagen.generate(32)
    probs = model(x)
    loss = F.binary_cross_entropy(probs.flatten(), label)
    loss.backward()
    if inject_all_reduce:
        inject_all_reduce()
    optim.step()
    optim.zero_grad(set_to_none=True)
