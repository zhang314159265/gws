def get_num_params(model):
    tot = 0

    for param in model.parameters():
        tot += param.numel()
    return tot
