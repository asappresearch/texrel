import torch


def get_rng_state(device):
    if device == 'cuda':
        # with torch.cuda.device(device):
        return torch.cuda.get_rng_state()
    elif device == 'cpu':
        return torch.get_rng_state()
    raise Exception('unknown device type', device)


def set_rng_state(device, state):
    if device == 'cuda':
        # with torch.cuda.device(device):
        torch.cuda.set_rng_state(state)
    elif device == 'cpu':
        torch.set_rng_state(state)
    else:
        raise Exception('unknown device type', device)


class TorchRandomState(object):
    """
    Note: initializing this DESTROYS any existing torch rng state...

    Since it's designed to be used systematically, this wont be an issue,
    IN THIS TARGET USE-CASE

    only handles single gpu for now

    device can be 'cuda' or 'cpu' (ie 'cuda:0' etc wont work)
    """
    def __init__(self, seed):
        # print('TorchRandomState warning ignoring device arg')
        # self.device = device
        self.seed = seed
        # old_state = get_rng_state(device)
        torch.manual_seed(seed)
        # self.cpu_state = get_rng_state(device='cpu')
        # self.cuda_state = get_rng_state(device='cuda')
        self.cpu_state = torch.get_rng_state()
        self.cuda_state = torch.cuda.get_rng_state()

    def __enter__(self):
        # set_rng_state(device=self.device, state=self.state)
        torch.set_rng_state(self.cpu_state)
        torch.cuda.set_rng_state(self.cuda_state)

    def __exit__(self, *args):
        # self.state = get_rng_state(device=self.device)
        self.cpu_state = torch.get_rng_state()
        self.cuda_state = torch.cuda.get_rng_state()
