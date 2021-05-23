from torch import nn
from ulfs import nn_modules


def add_layers_from_netstring(module, net_string, in_channels, in_width, in_height):
    """
    eg c:8,3 r mp:2 c:16,3 r mp:2 flat fc:{embedding_size}
    """
    # builder = LayerBuilder(module)
    builder = nn.ModuleList()
    channels = in_channels
    width = in_width
    height = in_height
    neurons = None
    for netbit in net_string.split(' '):
        name = netbit.split(':')[0]
        params = []
        if len(netbit.split(':')) > 1:
            params = netbit.split(':')[1].split(',')
            params_new = []
            for param in params:
                # params_new.append(int(param.format(embedding_size=embedding_size)))
                params_new.append(int(param))
            params = params_new
        if name == 'c':
            builder.append(nn.Conv2d(
                in_channels=channels, out_channels=params[0], kernel_size=params[1], padding=(params[1] // 2)))
            channels = params[0]
        elif name in ['r', 'relu']:
            builder.append(nn.ReLU())
        elif name == 'tanh':
            builder.append(nn.Tanh())
        elif name == 'mp':
            builder.append(nn.MaxPool2d(params[0]))
            width = width // params[0]
            height = height // params[0]
        elif name == 'flat':
            neurons = width * height * channels
            builder.append(nn_modules.Flatten())
        elif name == 'fc':
            # if is_2d:
            print('neurons', neurons, 'params[0]', params[0])
            builder.append(nn.Linear(neurons, params[0]))
            neurons = params[0]
        else:
            raise Exception('unknown layer', name)
    return builder
    # self.layers = builder.layers
