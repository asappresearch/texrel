import math
from typing import Dict, Type

import torch
from torch import nn

from ulfs.tensor_utils import Hadamard, concat


cells_by_name: Dict[str, Type[nn.Module]] = {}


# def register(name, cls):
#     def _decorator(func):
#         return func
#     print('register', name)
#     cells_by_name[name] = func
#     return _decorator


class MyLSTMCell_concatfused(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        # assert input_size == hidden_size
        self.input_size = input_size
        self.embedding_size = hidden_size
        # self.h1 = nn.Linear(self.embedding_size * 2, self.embedding_size * 4)
        self.h1 = nn.Linear(self.input_size + self.embedding_size, self.embedding_size * 4)

    def forward(self, x, state_cell_tuple):
        state, cell = state_cell_tuple
        batch_size = x.size()[0]
        in_concat = concat(x, state)
        xdot = self.h1(in_concat)
        xdot = xdot.view(batch_size, 4, self.embedding_size)
        i = torch.tanh(xdot[:, 0])
        j = torch.sigmoid(xdot[:, 1])
        f = torch.sigmoid(xdot[:, 2])
        o = torch.tanh(xdot[:, 3])
        celldot = Hadamard(cell, f) + Hadamard(i, j)
        statedot = Hadamard(torch.tanh(celldot), o)
        return (statedot, celldot)


cells_by_name['mylstm_concatfused'] = MyLSTMCell_concatfused


class MyLSTMCell_concat(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.input_size = input_size
        self.embedding_size = hidden_size
        self.h1 = nn.Linear(self.input_size + self.embedding_size, self.embedding_size)
        self.h2 = nn.Linear(self.input_size + self.embedding_size, self.embedding_size)
        self.h3 = nn.Linear(self.input_size + self.embedding_size, self.embedding_size)
        self.h4 = nn.Linear(self.input_size + self.embedding_size, self.embedding_size)

    def forward(self, x, state_cell_tuple):
        state, cell = state_cell_tuple
        in_concat = concat(x, state)
        # xdot = self.h1(in_concat)
        # xdot = xdot.view(batch_size, 4, self.embedding_size)
        i = torch.tanh(self.h1(in_concat))
        j = torch.sigmoid(self.h2(in_concat))
        f = torch.sigmoid(self.h3(in_concat))
        o = torch.tanh(self.h4(in_concat))
        celldot = Hadamard(cell, f) + Hadamard(i, j)
        statedot = Hadamard(torch.tanh(celldot), o)
        return (statedot, celldot)


cells_by_name['mylstm_concat'] = MyLSTMCell_concat


class MyLSTMCell_concatgrouped(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.input_size = input_size
        self.embedding_size = hidden_size
        self.h1a = nn.Linear(self.input_size // 2 + self.embedding_size // 2, self.embedding_size)
        self.h2a = nn.Linear(self.input_size // 2 + self.embedding_size // 2, self.embedding_size)
        self.h3a = nn.Linear(self.input_size // 2 + self.embedding_size // 2, self.embedding_size)
        self.h4a = nn.Linear(self.input_size // 2 + self.embedding_size // 2, self.embedding_size)

        self.h1b = nn.Linear(self.input_size // 2 + self.embedding_size // 2, self.embedding_size)
        self.h2b = nn.Linear(self.input_size // 2 + self.embedding_size // 2, self.embedding_size)
        self.h3b = nn.Linear(self.input_size // 2 + self.embedding_size // 2, self.embedding_size)
        self.h4b = nn.Linear(self.input_size // 2 + self.embedding_size // 2, self.embedding_size)

    def forward(self, x, state_cell_tuple):
        state, cell = state_cell_tuple
        x1 = x[:, :self.input_size//2]
        x2 = x[:, self.input_size//2:]
        h1 = state[:, :self.embedding_size//2]
        h2 = state[:, self.embedding_size//2:]
        in_concat1 = concat(x1, h1)
        in_concat2 = concat(x2, h2)
        # in_concat = concat(x, state)
        # xdot = self.h1(in_concat)
        # xdot = xdot.view(batch_size, 4, self.embedding_size)
        i = torch.tanh(self.h1a(in_concat1) + self.h1b(in_concat2))
        j = torch.sigmoid(self.h2a(in_concat1) + self.h2b(in_concat2))
        f = torch.sigmoid(self.h3a(in_concat1) + self.h3b(in_concat2))
        o = torch.tanh(self.h4a(in_concat1) + self.h4b(in_concat2))
        celldot = Hadamard(cell, f) + Hadamard(i, j)
        statedot = Hadamard(torch.tanh(celldot), o)
        return (statedot, celldot)


cells_by_name['mylstm_concatgrouped'] = MyLSTMCell_concatgrouped


class MyLSTMCellFused(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.input_size = input_size
        self.embedding_size = hidden_size

        self.Wx = nn.Parameter(torch.Tensor(input_size, 4 * hidden_size))
        self.Wh = nn.Parameter(torch.Tensor(hidden_size, 4 * hidden_size))

        self.bias = nn.Parameter(torch.Tensor(4, hidden_size))
        # this initialization adapted from pytorch nn.Linear
        stdv = 1. / math.sqrt(hidden_size)
        self.bias.data.uniform_(-stdv, stdv)
        self.Wx.data.uniform_(-stdv, stdv)
        self.Wh.data.uniform_(-stdv, stdv)

    def forward(self, x, state_cell_tuple):
        state, cell = state_cell_tuple
        batch_size = x.size()[0]

        xdot = x @ self.Wx
        xdot = xdot.view(batch_size, 4, self.embedding_size)
        hdot = state @ self.Wh
        hdot = hdot.view(batch_size, 4, self.embedding_size)

        i = torch.tanh(xdot[:, 0] + hdot[:, 0] + self.bias[0])
        j = torch.sigmoid(xdot[:, 1] + hdot[:, 1] + self.bias[1])
        f = torch.sigmoid(xdot[:, 2] + hdot[:, 2] + self.bias[2])
        o = torch.tanh(xdot[:, 3] + hdot[:, 3] + self.bias[3])

        celldot = Hadamard(cell, f) + Hadamard(i, j)
        statedot = Hadamard(torch.tanh(celldot), o)
        return (statedot, celldot)


cells_by_name['mylstmfused'] = MyLSTMCellFused


class MyLSTMCell(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.input_size = input_size
        self.embedding_size = hidden_size

        self.fc_x1 = nn.Linear(self.input_size, self.embedding_size)
        self.fc_x2 = nn.Linear(self.input_size, self.embedding_size)
        self.fc_x3 = nn.Linear(self.input_size, self.embedding_size)
        self.fc_x4 = nn.Linear(self.input_size, self.embedding_size)
        self.fc_h1 = nn.Linear(self.embedding_size, self.embedding_size)
        self.fc_h2 = nn.Linear(self.embedding_size, self.embedding_size)
        self.fc_h3 = nn.Linear(self.embedding_size, self.embedding_size)
        self.fc_h4 = nn.Linear(self.embedding_size, self.embedding_size)

    def forward(self, x, state_cell_tuple):
        state, cell = state_cell_tuple

        i = torch.tanh(self.fc_x1(x) + self.fc_h1(state))
        j = torch.sigmoid(self.fc_x2(x) + self.fc_h2(state))
        f = torch.sigmoid(self.fc_x3(x) + self.fc_h3(state))
        o = torch.tanh(self.fc_x4(x) + self.fc_h4(state))

        celldot = Hadamard(cell, f) + Hadamard(i, j)
        statedot = Hadamard(torch.tanh(celldot), o)
        return (statedot, celldot)


cells_by_name['mylstm'] = MyLSTMCell


class NegOut(nn.Module):
    def __init__(self, prob):
        super().__init__()
        self.prob = prob
        self._train = True

    def forward(self, x):
        # print('dir(self)', dir(self))
        # print('self.train', self.train)
        # asdf
        if not self._train:
            return x
        shape = x.size()
        self.mask = 1 - 2 * (torch.rand(*shape) < self.prob).int()
        # self.mask = 1 - 2 * (torch.rand((1, shape[-1])).expand(*list(shape)) < self.prob).int()
        x = x * self.mask.float()
        return x

    def train(self, arg):
        # print('train', arg)
        super().train(arg)
        # if arg != self._train:
        #     print('train change', arg)
        self._train = arg


class MyNegoutGRUv3Cell(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.input_size = input_size
        self.embedding_size = hidden_size

        self.fc_x1 = nn.Linear(self.input_size, self.embedding_size)
        self.fc_x2 = nn.Linear(self.input_size, self.embedding_size)
        self.fc_x3 = nn.Linear(self.input_size, self.embedding_size)
        self.fc_h1 = nn.Linear(self.embedding_size, self.embedding_size)
        self.fc_h2 = nn.Linear(self.embedding_size, self.embedding_size)

        self.fc_rh = nn.Linear(self.embedding_size, self.embedding_size)

        self.negout = NegOut(0.1)

    def forward(self, x, state):
        # state = self.negout(state)
        # x = self.negout(x)

        r = torch.sigmoid(self.negout(self.fc_x1(x)) + self.fc_h1(state))
        z = torch.sigmoid(self.negout(self.fc_x2(x)) + self.fc_h2(state))
        htilde = torch.tanh(
            self.negout(self.fc_x3(x)) + self.fc_rh(Hadamard(r, state)))
        hdot = Hadamard(z, state) + Hadamard(1 - z, htilde)
        return hdot


class MyGRUv3Cell(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.input_size = input_size
        self.embedding_size = hidden_size

        self.fc_x1 = nn.Linear(self.input_size, self.embedding_size)
        self.fc_x2 = nn.Linear(self.input_size, self.embedding_size)
        self.fc_x3 = nn.Linear(self.input_size, self.embedding_size)
        self.fc_h1 = nn.Linear(self.embedding_size, self.embedding_size)
        self.fc_h2 = nn.Linear(self.embedding_size, self.embedding_size)

        self.fc_rh = nn.Linear(self.embedding_size, self.embedding_size)

    def forward(self, x, state):
        r = torch.sigmoid(self.fc_x1(x) + self.fc_h1(state))
        z = torch.sigmoid(self.fc_x2(x) + self.fc_h2(state))
        htilde = torch.tanh(
            self.fc_x3(x) + self.fc_rh(Hadamard(r, state)))
        hdot = Hadamard(z, state) + Hadamard(1 - z, htilde)
        return hdot


class MyNoisyGRUv3Cell(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.input_size = input_size
        self.embedding_size = hidden_size

        self.fc_x1 = nn.Linear(self.input_size, self.embedding_size)
        self.fc_x2 = nn.Linear(self.input_size, self.embedding_size)
        self.fc_x3 = nn.Linear(self.input_size, self.embedding_size)
        self.fc_h1 = nn.Linear(self.embedding_size, self.embedding_size)
        self.fc_h2 = nn.Linear(self.embedding_size, self.embedding_size)

        self.fc_rh = nn.Linear(self.embedding_size, self.embedding_size)

    def get_noise(self, batch_size):
        # return torch.randn(self.embedding_size).expand_as(batch_size, self.embedding_size) * 0.1
        return torch.randn(batch_size, self.embedding_size) * 0.1

    def forward(self, x, state):
        r = torch.sigmoid(self.fc_x1(x) + self.fc_h1(state))
        z = torch.sigmoid(self.fc_x2(x) + self.fc_h2(state))
        htilde = torch.tanh(
            self.fc_x3(x) + self.fc_rh(Hadamard(r, state)))
        hdot = Hadamard(z, state) + Hadamard(1 - z, htilde)
        return hdot


class MyUGRUCell(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.input_size = input_size
        self.embedding_size = hidden_size

        self.fc_x1 = nn.Linear(self.input_size, self.embedding_size)
        self.fc_x2 = nn.Linear(self.input_size, self.embedding_size)
        self.fc_x3 = nn.Linear(self.input_size, self.embedding_size)
        self.fc_x4 = nn.Linear(self.input_size, self.embedding_size)
        self.fc_h1 = nn.Linear(self.embedding_size, self.embedding_size)
        self.fc_h2 = nn.Linear(self.embedding_size, self.embedding_size)
        self.fc_h4 = nn.Linear(self.embedding_size, self.embedding_size)

        self.fc_rh = nn.Linear(self.embedding_size, self.embedding_size)

    def forward(self, x, state):
        self.r = torch.sigmoid(self.fc_x1(x) + self.fc_h1(state))
        self.z = torch.sigmoid(self.fc_x2(x) + self.fc_h2(state))
        self.i = torch.sigmoid(self.fc_x4(x) + self.fc_h4(state))
        self.htilde = torch.tanh(
            self.fc_x3(x) + self.fc_rh(Hadamard(self.r, state)))
        hdot = Hadamard(self.z, state) + Hadamard(self.i, self.htilde)
        return hdot


class MyGRUv3Cellz(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.input_size = input_size
        self.embedding_size = hidden_size

        self.fc_x1 = nn.Linear(self.input_size, self.embedding_size)
        self.fc_x2 = nn.Linear(self.input_size, self.embedding_size)
        self.fc_x3 = nn.Linear(self.input_size, self.embedding_size)
        self.fc_h1 = nn.Linear(self.embedding_size, self.embedding_size)
        self.fc_h2 = nn.Linear(self.embedding_size, self.embedding_size)

        self.fc_rh = nn.Linear(self.embedding_size, self.embedding_size)

    def forward(self, x, state):
        r = torch.sigmoid(self.fc_x1(x) + self.fc_h1(state))
        z = torch.sigmoid(self.fc_x2(x) + self.fc_h2(state))
        htilde = torch.tanh(
            self.fc_x3(x) + self.fc_rh(Hadamard(r, state)))
        hdot = Hadamard(1 - z, state) + Hadamard(z, htilde)
        return hdot


class MyGRUv1Cell(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.input_size = input_size
        self.embedding_size = hidden_size

        self.fc_x1 = nn.Linear(self.input_size, self.embedding_size)
        self.fc_x2 = nn.Linear(self.input_size, self.embedding_size)
        self.fc_x3 = nn.Linear(self.input_size, self.embedding_size)
        self.fc_h1 = nn.Linear(self.embedding_size, self.embedding_size)
        self.fc_h2 = nn.Linear(self.embedding_size, self.embedding_size)
        self.fc_h3 = nn.Linear(self.embedding_size, self.embedding_size)

    def forward(self, x, state):
        r = torch.sigmoid(self.fc_x1(x) + self.fc_h1(state))
        z = torch.sigmoid(self.fc_x2(x) + self.fc_h2(state))
        n = torch.tanh(self.fc_x3(x) + Hadamard(r, self.fc_h3(state)))
        hdot = Hadamard(1 - z, n) + Hadamard(z, state)
        return hdot


for k, v in {
    'mygruv3': MyGRUv3Cell,
    'mygruv1': MyGRUv1Cell,
    'mygruv3z': MyGRUv3Cellz,
    'ugru': MyUGRUCell,
    'noisygruv3': MyNoisyGRUv3Cell,
    'mynegoutgru': MyNegoutGRUv3Cell
}.items():
    cells_by_name[k] = v
