import time

import pytest
import torch
from torch import nn, optim
import numpy as np

from ulfs import nn_modules
from ulfs import tensor_utils


def test_spatial_to_vector():
    spatial_to_vector = nn_modules.SpatialToVector()
    M = 5
    N = 32
    C = 8
    H = 5
    W = 5
    a = torch.rand(M, N, C, H, W)
    b = spatial_to_vector(a)
    print('b.size()', b.size())
    assert len(b.size()) == 3
    assert b.size(0) == 5
    assert b.size(1) == 32
    assert b.size(2) == 8 * 5 * 5


def test_multi_embedding():
    batch_size = 5
    embedding_size = 4
    print('batch_size', batch_size)
    print('embedding_size', embedding_size)
    sizes_l = [2, 3, 5]
    model = nn_modules.MultiEmbedding(sizes_l=sizes_l, embedding_size=embedding_size)
    input1 = torch.from_numpy(np.random.choice(2, (batch_size, 1), replace=True)).long()
    input2 = torch.from_numpy(np.random.choice(3, (batch_size, 1), replace=True)).long()
    input3 = torch.from_numpy(np.random.choice(5, (batch_size, 1), replace=True)).long()
    input = torch.cat([input1, input2, input3], dim=1)
    print('input', input)
    print('input.size()', input.size(), input.dtype)

    out = model(input)
    print('out', out)
    print('out.size()', out.size())


@pytest.mark.skip
def test_multi_embedding_autoenc():
    batch_size = 5
    sizes_l = [3, 2, 5]

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.embedding = nn_modules.MultiEmbedding(sizes_l=sizes_l, embedding_size=np.sum(sizes_l))

        def forward(self, x):
            x = self.embedding(x)
            return x

    model = Model()
    opt = optim.Adam(lr=0.001, params=model.parameters())
    episode = 0
    crit = nn.BCEWithLogitsLoss()
    last_print = time.time()
    while True:
        X = torch.LongTensor(batch_size, len(sizes_l)).zero_()
        for i, size in enumerate(sizes_l):
            X[:, i] = torch.from_numpy(np.random.choice(size, batch_size, replace=True))
        out = model(X)
        pred = tensor_utils.multi_argmax(sizes_l=sizes_l, tensor=out)
        X_onehot = tensor_utils.multi_onehot(sizes_l=sizes_l, tensor=X)
        loss = crit(out, X_onehot)
        if time.time() - last_print > 0.5:
            print('diff', X - pred)
            print('episode', episode, 'loss', loss.item())
            last_print = time.time()
            if episode > 2500:
                break
        opt.zero_grad()
        loss.backward()
        opt.step()
        episode += 1
    assert (X - pred).abs().sum().item() == 0


def test_rcnn_cell():
    class CNNModel(nn.Module):
        def __init__(self, grid_planes, dropout, cnn_sizes):
            super().__init__()
            self.cnn_sizes = cnn_sizes
            self.cnn_layers = nn.ModuleList()
            last_channels = grid_planes
            for i, cnn_size in enumerate(cnn_sizes):
                # note: no maxpooling, for better or worse...
                if i != 0:
                    self.cnn_layers.append(nn.Dropout(dropout))
                    self.cnn_layers.append(nn.ReLU())
                self.cnn_layers.append(
                    nn.Conv2d(in_channels=last_channels, out_channels=cnn_size, kernel_size=3, padding=1))
                last_channels = cnn_size

        def forward(self, grids):
            for layer in self.cnn_layers:
                grids = layer(grids)
            return grids

    input_sequences = torch.LongTensor([
        [
            [[1, 0],
             [0, 2]],
            [[0, 2],
             [1, 0]]
        ],
        [
            [[2, 0],
             [0, 1]],
            [[0, 2],
             [1, 0]]
        ]
    ])
    labels = torch.LongTensor([0, 1])
    print('input_sequences', input_sequences)
    print('input_sequences.size()', input_sequences.size())

    # input_planes = 3
    # hidden_planes = 4
    # grid_size = 5

    batch_size = input_sequences.size()[0]
    seq_len = input_sequences.size()[1]
    input_planes = input_sequences.max().item() + 1
    # input_planes = input_sequences.size()[2]
    grid_size = input_sequences.size()[2]

    input_sequences_fluffy = torch.Tensor(batch_size, seq_len, input_planes, grid_size, grid_size).zero_()
    for n in range(batch_size):
        for t in range(seq_len):
            for h in range(grid_size):
                for w in range(grid_size):
                    color_id = input_sequences[n][t][h][w]
                    input_sequences_fluffy[n][t][color_id][h][w] = 1
    print('input_sequences_fluffy', input_sequences_fluffy)

    def cnn_constr(input_planes, output_planes):
        cnn = CNNModel(grid_planes=input_planes, dropout=0.5, cnn_sizes=[output_planes])
        return cnn

    class Model(nn.Module):
        def __init__(self, input_planes, hidden_planes, grid_size):
            super().__init__()
            self.input_planes = input_planes
            self.hidden_planes = hidden_planes
            self.grid_size = grid_size
            self.rcnn = nn_modules.RCNNCell(
                cnn_constr=cnn_constr,
                input_planes=input_planes,
                hidden_planes=hidden_planes
            )
            self.flatten = nn_modules.Flatten()
            self.h1 = nn.Linear(hidden_planes * grid_size * grid_size, 2)

        def forward(self, x):
            batch_size = x.size()[0]
            seq_len = x.size()[1]
            state = torch.Tensor(batch_size, self.hidden_planes, self.grid_size, self.grid_size).zero_()
            cell = torch.Tensor(batch_size, self.hidden_planes, self.grid_size, self.grid_size).zero_()

            for t in range(seq_len):
                state, cell = self.rcnn(x[:, t], (state, cell))
            x = self.flatten(state)
            x = self.h1(x)
            return x

    model = Model(input_planes=input_planes, hidden_planes=input_planes, grid_size=grid_size)
    opt = optim.Adam(lr=0.001, params=model.parameters())
    crit = nn.CrossEntropyLoss()
    episode = 0
    last_print = time.time()
    while True:
        out = model(input_sequences_fluffy)
        _, pred = out.max(dim=-1)
        loss = crit(out, labels)
        opt.zero_grad()
        loss.backward()
        opt.step()
        if time.time() - last_print >= 1.0:
            print('pred', pred)
            print('pred.size()', pred.size())
            print('episode', episode, 'loss', loss.item())
            last_print = time.time()
            if episode >= 500:
                break
        episode += 1
    assert pred[0].item() == 0
    assert pred[1].item() == 1


def test_rcnn_multi():
    # class CNNModel(nn.Module):
    #     def __init__(self, grid_planes, dropout, cnn_sizes):
    #         super().__init__()
    #         self.cnn_sizes = cnn_sizes
    #         self.cnn_layers = nn.ModuleList()
    #         last_channels = grid_planes
    #         for i, cnn_size in enumerate(cnn_sizes):
    #             # note: no maxpooling, for better or worse...
    #             if i != 0:
    #                 self.cnn_layers.append(nn.Dropout(dropout))
    #                 self.cnn_layers.append(nn.ReLU())
    #             self.cnn_layers.append(
    #                 nn.Conv2d(in_channels=last_channels, out_channels=cnn_size, kernel_size=3, padding=1))
    #             last_channels = cnn_size

    #     def forward(self, grids):
    #         for l in self.cnn_layers:
    #             grids = l(grids)
    #         return grids

    input_sequences = torch.LongTensor([
        [
            [[1, 0],
             [0, 2]],
            [[0, 2],
             [1, 0]]
        ],
        [
            [[2, 0],
             [0, 1]],
            [[0, 2],
             [1, 0]]
        ]
    ])
    labels = torch.LongTensor([0, 1])
    print('input_sequences', input_sequences)
    print('input_sequences.size()', input_sequences.size())

    # input_planes = 3
    # hidden_planes = 4
    # grid_size = 5

    batch_size = input_sequences.size()[0]
    seq_len = input_sequences.size()[1]
    input_planes = input_sequences.max().item() + 1
    # input_planes = input_sequences.size()[2]
    grid_size = input_sequences.size()[2]

    input_sequences_fluffy = torch.Tensor(batch_size, seq_len, input_planes, grid_size, grid_size).zero_()
    for n in range(batch_size):
        for t in range(seq_len):
            for h in range(grid_size):
                for w in range(grid_size):
                    color_id = input_sequences[n][t][h][w]
                    input_sequences_fluffy[n][t][color_id][h][w] = 1
    print('input_sequences_fluffy', input_sequences_fluffy)

    def cnn_constr(input_channels, output_channels):
        cnn = nn.Conv2d(in_channels=input_channels, out_channels=output_channels, kernel_size=3, padding=1)
        # cnn = CNNModel(grid_planes=input_planes, dropout=0.5, cnn_sizes=[output_planes])
        return cnn

    class Model(nn.Module):
        def __init__(self, input_planes, hidden_planes, grid_size):
            super().__init__()
            self.input_planes = input_planes
            self.hidden_planes = hidden_planes
            self.grid_size = grid_size
            self.rcnn = nn_modules.RCNN(
                cnn_constr=cnn_constr,
                input_planes=input_planes,
                hidden_planes=hidden_planes,
                grid_size=grid_size,
                dropout=0.5,
                num_layers=2
            )
            self.flatten = nn_modules.Flatten()
            self.h1 = nn.Linear(hidden_planes * grid_size * grid_size, 2)

        def forward(self, x):
            x = x.transpose(0, 1).contiguous()

            # state = torch.Tensor(batch_size, self.hidden_planes, self.grid_size, self.grid_size).zero_()
            # cell = torch.Tensor(batch_size, self.hidden_planes, self.grid_size, self.grid_size).zero_()

            out, (states, cells) = self.rcnn(x)
            # for t in range(seq_len):
            #     state, cell = self.rcnn(x[:, t], (state, cell))
            x = self.flatten(out[-1])
            x = self.h1(x)
            return x

    model = Model(input_planes=input_planes, hidden_planes=input_planes, grid_size=grid_size)
    opt = optim.Adam(lr=0.001, params=model.parameters())
    crit = nn.CrossEntropyLoss()
    episode = 0
    last_print = time.time()
    while True:
        out = model(input_sequences_fluffy)
        _, pred = out.max(dim=-1)
        loss = crit(out, labels)
        opt.zero_grad()
        loss.backward()
        opt.step()
        if time.time() - last_print >= 1.0:
            print('pred', pred)
            print('pred.size()', pred.size())
            print('episode', episode, 'loss', loss.item())
            last_print = time.time()
            if episode >= 500:
                break
        episode += 1
    assert pred[0].item() == 0
    assert pred[1].item() == 1


def test_masked_crit_bce():
    unmasked_crit = nn.BCEWithLogitsLoss()
    masked_crit = nn_modules.MaskedCrit(nn.BCEWithLogitsLoss)

    N = 3
    K = 4
    pred = torch.rand(N, K)
    tgt = torch.rand(N, K)
    mask = (torch.rand(N, K) > 0.5).bool()
    # print('', a)
    # print('b', b)

    loss_unmasked = unmasked_crit(pred, tgt)
    loss_masked = masked_crit(pred, tgt, mask=mask)
    print('loss_unmasked', loss_unmasked.item())
    print('loss_masked', loss_masked.item())

    mask.fill_(0)
    loss_masked = masked_crit(pred, tgt, mask=mask)
    print('loss_masked fill 0', loss_masked.item())

    mask.fill_(1)
    loss_masked = masked_crit(pred, tgt, mask=mask)
    print('loss_masked fill 1', loss_masked.item())


def test_masked_crit_ce():
    unmasked_crit = nn.CrossEntropyLoss()
    masked_crit = nn_modules.MaskedCrit(nn.CrossEntropyLoss)

    N = 3
    num_classes = 4
    pred = torch.rand(N, num_classes)
    # tgt = torch.rand(N, K)
    tgt = (torch.rand(N) * num_classes).long()
    mask = (torch.rand(N) > 0.5).bool()
    # print('', a)
    # print('b', b)

    print('pred.size()', pred.size())
    print('tgt.size()', tgt.size())
    loss_unmasked = unmasked_crit(pred, tgt)
    loss_masked = masked_crit(pred, tgt, mask=mask)
    print('loss_unmasked', loss_unmasked.item())
    print('loss_masked', loss_masked.item())

    mask.fill_(0)
    loss_masked = masked_crit(pred, tgt, mask=mask)
    print('loss_masked fill 0', loss_masked.item())

    mask.fill_(1)
    loss_masked = masked_crit(pred, tgt, mask=mask)
    print('loss_masked fill 1', loss_masked.item())


def test_flattened_ce_loss_reduce():
    crit = nn_modules.GeneralCrossEntropyLoss()
    a = torch.rand(2, 3, 5)
    b = (torch.rand(2, 3) * 5).long()
    loss = crit(a, b)
    print('loss', loss.item())


def test_flattened_ce_loss_no_reduce():
    crit = nn_modules.GeneralCrossEntropyLoss(reduction='none')
    a = torch.rand(2, 3, 5)
    b = (torch.rand(2, 3) * 5).long()
    loss = crit(a, b)
    print('loss', loss)
    print('loss.size()', loss.size())


def test_masked_acc():
    N = 64
    num_classes = 4
    # pred = torch.rand(N, num_classes)
    pred = (torch.rand(4, N) * num_classes).long()
    tgt = (torch.rand(4, N) * num_classes).long()
    mask = (torch.rand(4, N) > 0.5).bool()

    masked_acc = nn_modules.MaskedAcc()
    acc = masked_acc(pred, tgt, mask)
    print('acc', acc)

    mask.fill_(0)
    acc = masked_acc(pred, tgt, mask)
    print('acc', acc)

    mask.fill_(1)
    acc = masked_acc(pred, tgt, mask)
    print('acc', acc)

    acc_mod = nn_modules.Acc()
    acc = acc_mod(pred, tgt)
    print('acc', acc)


def test_pointwise_conv3d():
    torch.manual_seed(123)
    N = 4
    H = 5
    W = 5
    Ci = 3
    Co = 6
    input = torch.rand(N, Ci, H, W)
    conv = nn_modules.PointwiseConv3d(Ci, Co)
    output = conv(input)
    print('output.size()', output.size())


def test_bahdenau_attention():
    N = 3
    E = 5
    Mi = 4
    print('')
    print('N', N, 'E', E, 'Mi', Mi)
    # Mo = 6

    torch.manual_seed(123)
    inputs = torch.rand(Mi, N, E)
    # last_output = torch.rand(N, E)
    last_output = torch.rand(N, E)

    att_model = nn_modules.BahdenauAttention(embedding_size=E)
    projected_keys = att_model.project_keys(keys=inputs)
    att = att_model.calc_attention(projected_keys=projected_keys, queries=last_output)
    print('att', att)
    out = att_model.apply_attention(att=att, values=inputs)
    print('out', out)
    print('out.size()', out.size())
    print('out.sum(0)', out.sum(0))
