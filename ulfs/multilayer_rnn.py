import torch
from torch import nn


class MyMultilayerRNN(nn.Module):
    def __init__(self, needs_cell, constr, input_size, hidden_size, num_layers, dropout, bidirectional):
        super().__init__()
        assert not bidirectional
        self.num_layers = num_layers
        self.needs_cell = needs_cell
        self.hidden_size = hidden_size

        self.module_list = nn.ModuleList()
        for i in range(num_layers):
            _input_size = input_size if i == 0 else hidden_size
            layer = constr(input_size=_input_size, hidden_size=hidden_size)
            self.module_list.append(layer)
            # if not last_layer and dropout > 0:
            #     l = nn.Dropout(dropout)
            #     self.module_list.append(l)
        self.dropout = dropout
        if dropout > 0:
            self.drop = nn.Dropout(dropout)

    def forward(self, packed, state_cell_tuple=None):
        x, x_len = torch.nn.utils.rnn.pad_packed_sequence(packed)
        # print('x.is_cuda', x.is_cuda)
        is_cuda = x.is_cuda

        assert len(x.size()) == 3
        seq_len = x.size()[0]
        batch_size = x.size()[1]

        if state_cell_tuple is not None:
            if self.needs_cell:
                in_states, in_cells = state_cell_tuple
                in_states = in_states.detach()
                in_cells = in_cells.detach()
            else:
                in_states = state_cell_tuple
                in_states = in_states.detach()
        else:
            in_states = torch.zeros(self.num_layers, batch_size, self.hidden_size)
            in_cells = torch.zeros(self.num_layers, batch_size, self.hidden_size)
            if is_cuda:
                in_states = in_states.cuda()
                in_cells = in_cells.cuda()
        # states = states.detach()
        # cells = cells.detach()

        out_states = torch.zeros(self.num_layers, batch_size, self.hidden_size)
        out_cells = torch.zeros(self.num_layers, batch_size, self.hidden_size)
        if is_cuda:
            out_states = out_states.cuda()
            out_cells = out_cells.cuda()

        states = []
        cells = []
        for i in range(self.num_layers):
            states.append(in_states[i])
            if self.needs_cell:
                cells.append(in_cells[i])

        # out_state = torch.zeros(batch_size, self.hidden_size)
        output = torch.zeros(seq_len, batch_size, self.hidden_size)
        if is_cuda:
            output = output.cuda()

        # debugging...
        # for i, l in enumerate(self.module_list):
        #     _s = torch.zeros(batch_size, self.hidden_size)
        #     _c = torch.zeros(batch_size, self.hidden_size)
        #     _s, _c = l(x[0], (_s, _c))
        #     print('i', i)
        #     # print('_o', _o)
        #     print('_s', _s.tolist())
        #     print('_c', _c.tolist())

        for t in range(seq_len):
            for i, l in enumerate(self.module_list):
                # if i > 0:
                #     continue  # debugging...
                # if is_cuda:
                #     l = l.cuda()
                if i == 0:
                    _input = x[t]
                else:
                    _input = states[i-1]

                # if not is_last_layer and self.dropout > 0:
                if i > 0 and self.dropout > 0:
                    _input = self.drop(_input)

                # print('_input.is_cuda', _input.is_cuda)
                # print('states[i].is_cuda', states[i].is_cuda)
                if self.needs_cell:
                    # print(f'states[{i}]', states[i])
                    # print(f'cells[{i}]', cells[i])
                    # print('_input', _input)
                    states[i], cells[i] = l(_input, (states[i], cells[i]))
                    # print(f'states[{i}]', states[i])
                    # print(f'cells[{i}]', cells[i])
                else:
                    states[i] = l(_input, states[i])

                terminated_mask = x_len == t + 1
                terminated_idxes = terminated_mask.nonzero().view(-1)

                out_states[i, terminated_idxes] = states[i][terminated_idxes]
                out_cells[i, terminated_idxes] = cells[i][terminated_idxes]

                # if is_last_layer:

                # out_states[i] = states[i]
                # out_cells[i] = cells[i]

            output[t] = states[-1]
        # out = out_state
        # out = out_states[-1]

        for layer_idx in range(self.num_layers):
            out_states[layer_idx] = states[layer_idx].detach()
            out_cells[layer_idx] = cells[layer_idx].detach()

        output_packed = torch.nn.utils.rnn.pack_padded_sequence(output, x_len)

        if self.needs_cell:
            return output_packed, (out_states, out_cells)
        else:
            return output_packed, out_states
