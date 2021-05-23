from torch import nn


def get_num_parameters(model: nn.Module):
    num_params = 0
    for params in model.parameters():
        num_params += params.numel()
    return num_params


def dump_parameter_sizes(model: nn.Module):
    num_params = 0
    for k, p in model.named_parameters():
        print(k, p.numel())
        num_params += p.numel()
    print('total', num_params)
