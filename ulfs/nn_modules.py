import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from ulfs import alive_sieve, tensor_utils, rl_common
from ulfs.tensor_utils import Hadamard


def gumbel_softmax(logits, tau, hard, eps):
    logits_shape = logits.size()
    logits = logits.contiguous().view(-1, logits_shape[-1])
    logits = F.gumbel_softmax(logits, tau=tau, hard=hard, eps=eps)
    logits = logits.view(*logits_shape)
    return logits


class EmbeddingAdapter(nn.Module):
    """
    uses an nn.Embedding for discrete input, and an nn.Linear for onehot input
    """
    def __init__(self, vocab_size, embedding_size):
        super().__init__()
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, embedding_size)
        self.d2e = nn.Linear(vocab_size, embedding_size, bias=False)
        self.d2e.weight.data = self.embedding.weight.transpose(0, 1)

    def forward(self, utts):
        if utts.dtype == torch.int64:
            embs = self.embedding(utts)
        else:
            embs = self.d2e(utts)
        return embs


class PointwiseConv3d(nn.Module):
    def __init__(self, in_channels, out_channels, bias=True):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.h1 = nn.Linear(in_channels, out_channels, bias=bias)

    def forward(self, x):
        """
        input assumed to be [N][C][H][W]
        """
        assert len(x.size()) == 4
        x = self.h1(x.transpose(1, 3)).transpose(1, 3)
        return x


class Acc(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, pred, target):
        if pred.size() != target.size():
            print('size mismatch', pred.size(), 'vs', target.size())
        assert pred.size() == target.size()
        numel = pred.numel()
        correct = (pred == target)
        correct = correct.long().sum()
        acc = correct.float() / numel
        return acc.item()


class MaskedAcc(nn.Module):
    """
    returns accuracy assuming target and mask
    match when identical
    """
    def __init__(self):
        super().__init__()

    def forward(self, pred, target, mask):
        if pred.size() != target.size():
            print('size mismatch', pred.size(), 'vs', target.size())
        assert pred.size() == target.size()
        assert pred.size() == mask.size()
        numel = mask.long().sum()
        correct = (pred == target)
        correct = tensor_utils.masked_get(correct, mask)
        correct = correct.long().sum()
        acc = correct.float() / numel.float()
        return acc.item()


class GeneralNLLLoss(nn.Module):
    """
    flattens, and runs crit on last dimension,
    then unflattens
    """
    def __init__(self, reduction='mean'):
        """
        reduction can be 'none','mean','sum'
        """
        super().__init__()
        self.reduction = reduction
        self.crit = nn.NLLLoss(reduction=reduction)

    def forward(self, pred, target):
        pred_shape = list(pred.size())
        target_shape = list(target.size())
        num_classes = pred_shape[-1]
        pred_flat = pred.view(-1, num_classes)
        target_flat = target.view(-1)
        loss_flat = self.crit(pred_flat, target_flat)
        if self.reduction != 'none':
            return loss_flat
        loss = loss_flat.view(target_shape)
        return loss


class GeneralCrossEntropyLoss(nn.Module):
    """
    flattens, and runs crit on last dimension,
    then unflattens
    """
    def __init__(self, reduction='mean'):
        super().__init__()
        self.reduction = reduction
        self.crit = nn.CrossEntropyLoss(reduction=reduction)

    def forward(self, pred, target):
        pred_shape = list(pred.size())
        target_shape = list(target.size())
        num_classes = pred_shape[-1]
        pred_flat = pred.view(-1, num_classes)
        target_flat = target.view(-1)
        loss_flat = self.crit(pred_flat, target_flat)
        if self.reduction != 'none':
            return loss_flat
        loss = loss_flat.view(target_shape)
        return loss


class MaskedCrit(nn.Module):
    def __init__(self, crit_constr):
        super().__init__()
        self.crit = crit_constr(reduction='none')

    def forward(self, pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor):
        assert mask.dtype == torch.bool
        loss = self.crit(pred, target)
        loss = tensor_utils.masked_get(loss, mask)
        num_elem = mask.long().sum().item()
        loss = loss.sum() / num_elem
        return loss


class DiscreteEncoder(nn.Module):
    """
    Encoder is non-stochastic, fully differentiable

    Takes language => embedding
    """
    def __init__(self, vocab_size, embedding_size, d2e_embedding=None, rnn_cell=None):
        super().__init__()
        self.vocab_size = vocab_size
        self.embedding_size = embedding_size

        if d2e_embedding is None:
            d2e_embedding = nn.Embedding(vocab_size, embedding_size)
        self.d2e_embedding = d2e_embedding
        if rnn_cell is None:
            rnn_cell = nn.GRUCell(embedding_size, embedding_size)
        self.rnn_cell = rnn_cell

    def forward(self, utterance, term_id=0):
        """
        input: letters  [M][N]   (indexes, long)
        output: embedding  [N][E]   (embedding, float)

        terminates when hits term_id
        """
        device = utterance.device

        utterance_max, batch_size = utterance.size()
        sieve = alive_sieve.AliveSieve(batch_size=batch_size, enable_cuda=False)
        state = torch.zeros(batch_size, self.embedding_size, device=device)
        output_state = state.clone()
        for t in range(utterance_max):
            emb = self.d2e_embedding(utterance[t])
            state = self.rnn_cell(emb, state)
            output_state[sieve.global_idxes] = state
            sieve.mark_dead(utterance[t] == term_id)
            if sieve.all_dead():
                break

            utterance = utterance[:, sieve.alive_idxes]
            state = state[sieve.alive_idxes]
            sieve.self_sieve_()
        return output_state


class OnehotEncoder(nn.Module):
    """
    Encoder is non-stochastic, fully differentiable

    Takes language => embedding
    """
    def __init__(self, vocab_size, embedding_size, d2e_linear=None, rnn_cell=None):
        super().__init__()
        self.vocab_size = vocab_size
        self.embedding_size = embedding_size

        if d2e_linear is None:
            d2e_linear = nn.Linear(vocab_size, embedding_size)
        self.d2e_linear = d2e_linear
        if rnn_cell is None:
            rnn_cell = nn.GRUCell(embedding_size, embedding_size)
        self.rnn_cell = rnn_cell

    def forward(self, utterance):
        """
        input: letters  [M][N]   (indexes, long)
        output: embedding  [N][E]   (embedding, float)

        doesnt terminate, just keeps going to seq_len
        """
        device = utterance.device

        utterance_max, batch_size, vocab_size = utterance.size()
        state = torch.zeros(batch_size, self.embedding_size, device=device)
        output_state = state.clone()
        for t in range(utterance_max):
            emb = self.d2e_linear(utterance[t])
            state = self.rnn_cell(emb, state)
        output_state = state
        return output_state


class DifferentiableDecoder(nn.Module):
    """
    embedding => language

    Output is one-hot type distributions over lettrs, rather than
    hard argmaxd letter indices
    Therefore differentiable

    should we gumbelize it? I'm not sure :(
    """
    def __init__(self, vocab_size, embedding_size, rnn_cell=None, e2d_linear=None, d2e_linear=None):
        super().__init__()
        self.vocab_size = vocab_size
        self.embedding_size = embedding_size

        self.e2d_linear = nn.Linear(self.embedding_size, self.vocab_size) if e2d_linear is None else e2d_linear
        self.d2e_linear = nn.Linear(
            self.vocab_size, self.embedding_size, bias=False) if d2e_linear is None else d2e_linear
        self.rnn_cell = nn.GRUCell(self.embedding_size, self.embedding_size) if rnn_cell is None else rnn_cell

    def forward(self, x, max_len, target_utts=None):
        """
        Input is a batch of embedding vectors  [N][E]
        We'll convert to distributions over letters    [M][N][V]

        ignores zero terminations, just keeps going till max_len, for all batch examples

        if target_utts provided, we use teacher_forcing for the previous token each
        time
        """
        device = x.device
        batch_size = x.size()[0]
        state = x

        last_token = torch.zeros(batch_size, self.vocab_size, device=device)
        utterance = torch.zeros(max_len, batch_size, self.vocab_size, device=device)
        for t in range(max_len):
            emb = self.d2e_linear(last_token)
            state = self.rnn_cell(emb, state)
            token_logits = self.e2d_linear(state)
            token_probs = F.softmax(token_logits, dim=-1)
            utterance[t] = token_probs
            if target_utts is None:
                last_token = token_probs
            else:
                last_token = torch.zeros(batch_size, self.vocab_size, device=device)
                last_token.scatter_(1, target_utts[t].view(batch_size, 1), 1.0)
        return utterance


class StochasticDecoder(nn.Module):
    def __init__(self, vocab_size, embedding_size, utterance_max, rnn_cell, e2d_linear, d2e_embedding):
        super().__init__()
        self.utterance_max = utterance_max
        self.vocab_size = vocab_size
        self.embedding_size = embedding_size

        self.rnn = rnn_cell
        self.e2d = e2d_linear
        self.d2e = d2e_embedding

    def forward(self, x, global_idxes):
        """
        input is: [N][E]

        Input is a batch of embedding vectors
        We'll convert to discrete letters
        """

        batch_size = x.size()[0]
        state = x
        global_idxes = global_idxes.clone()
        # note that this sieve might start off smaller than the global batch_size
        sieve = alive_sieve.AliveSieve(batch_size=batch_size, enable_cuda=False)
        last_token = torch.zeros(batch_size, device=x.device, dtype=torch.int64)
        utterance = torch.zeros(batch_size, self.utterance_max, dtype=torch.int64, device=x.device)
        N_outer = torch.zeros(batch_size, device=x.device, dtype=torch.int64).fill_(self.utterance_max)

        stochastic_trajectory = rl_common.StochasticTrajectory()
        for t in range(self.utterance_max):
            emb = self.d2e(last_token)
            state = self.rnn(emb, state)
            token_logits = self.e2d(state)
            token_probs = F.softmax(token_logits, dim=-1)

            s = rl_common.draw_categorical_sample(
                action_probs=token_probs, batch_idxes=global_idxes[sieve.global_idxes])
            stochastic_trajectory.append_stochastic_sample(s=s)

            token = s.actions.view(-1)
            utterance[:, t][sieve.global_idxes] = token
            last_token = token
            sieve.mark_dead(last_token == 0)
            sieve.set_global_dead(N_outer, t + 1)
            if sieve.all_dead():
                break
            state = state[sieve.alive_idxes]
            last_token = last_token[sieve.alive_idxes]
            sieve.self_sieve_()
        res = {
            'stochastic_trajectory': stochastic_trajectory,
            'utterances': utterance,
            'utterances_lens': N_outer
        }
        return res


class Flatten(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        x = x.view(x.size()[0], -1)
        return x


class SpatialToVector(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        old_size = list(x.size())
        lhs = old_size[:-3]
        spatial_part = old_size[-3:]
        flat = np.prod(spatial_part).item()
        x = x.view(*(lhs + [flat]))
        return x


class MultiEmbedding(nn.Module):
    """
    we pass in a matrix with K columns, and sizes, each column contains indexes,
    into separate embeddings. The resulting embeddings are summed
    """
    def __init__(self, sizes_l, embedding_size):
        super().__init__()
        self.sizes_l = sizes_l
        self.embedding_size = embedding_size
        self.K = len(sizes_l)

        self.embeddings = nn.ModuleList()
        for k, size in enumerate(sizes_l):
            embedding = nn.Embedding(size, embedding_size)
            self.embeddings.append(embedding)

    def forward(self, x):
        assert len(x.size()) == 2

        res = None
        for k, size in enumerate(self.sizes_l):
            this_emb = self.embeddings[k](x[:, k])
            if res is None:
                res = this_emb
            else:
                res = res + this_emb
        return res


class RCNN(nn.Module):
    """
    Assumes fixed-size within-batch seq lens (can vary between batches)
    """
    def __init__(self, cnn_constr, input_planes, hidden_planes, grid_size, num_layers, dropout):
        super().__init__()

        self.droput = dropout
        self.num_layers = num_layers
        self.input_planes = input_planes
        self.hidden_planes = hidden_planes
        self.grid_size = grid_size

        self.layers = nn.ModuleList()
        for i in range(num_layers):
            rcnn = RCNNCell(
                cnn_constr=cnn_constr,
                input_planes=input_planes,
                hidden_planes=hidden_planes
            )
            self.layers.append(rcnn)
            input_planes = hidden_planes
        self.drop = nn.Dropout(dropout)

    def forward(self, grids):
        """
        grids is a [T][[N][C][H][W]  (where [T] is number of timesteps)
        """
        batch_size = grids.size()[1]
        seq_len = grids.size()[0]
        torch_constr = torch.cuda if grids.is_cuda else torch

        state_l = []
        cell_l = []
        out_states_l = []
        for i in range(self.num_layers):
            state_l.append(torch_constr.FloatTensor(
                batch_size, self.hidden_planes, self.grid_size, self.grid_size).zero_())
            cell_l.append(torch_constr.FloatTensor(
                batch_size, self.hidden_planes, self.grid_size, self.grid_size).zero_())

        for t in range(seq_len):
            for i, l in enumerate(self.layers):
                if i == 0:
                    x = grids[t]
                else:
                    x = state_l[i - 1]
                    x = self.drop(x)
                state_l[i], cell_l[i] = l(x, (state_l[i], cell_l[i]))
                if i == self.num_layers - 1:
                    out_states_l.append(state_l[i])
        out = torch.stack(out_states_l, dim=0)
        states = torch.stack(state_l, dim=0)
        cells = torch.stack(cell_l, dim=0)
        return out, (states, cells)


class RCNNCell(nn.Module):
    def __init__(self, cnn_constr, input_planes, hidden_planes):
        super().__init__()
        self.cnn_constr = cnn_constr

        self.h_x1 = cnn_constr(input_planes, hidden_planes)
        self.h_x2 = cnn_constr(input_planes, hidden_planes)
        self.h_x3 = cnn_constr(input_planes, hidden_planes)
        self.h_x4 = cnn_constr(input_planes, hidden_planes)
        self.h_h1 = cnn_constr(hidden_planes, hidden_planes)
        self.h_h2 = cnn_constr(hidden_planes, hidden_planes)
        self.h_h3 = cnn_constr(hidden_planes, hidden_planes)
        self.h_h4 = cnn_constr(hidden_planes, hidden_planes)

    def forward(self, x, state_cell_tuple):
        """
        x should be [N][C][H][W]  (where H == W)
        """
        state, cell = state_cell_tuple

        i = torch.tanh(self.h_x1(x) + self.h_h1(state))
        j = torch.sigmoid(self.h_x2(x) + self.h_h2(state))
        f = torch.sigmoid(self.h_x3(x) + self.h_h3(state))
        o = torch.tanh(self.h_x4(x) + self.h_h4(state))

        celldot = Hadamard(cell, f) + Hadamard(i, j)
        statedot = Hadamard(torch.tanh(celldot), o)
        return (statedot, celldot)


class BahdenauAttention(nn.Module):
    """
    """
    def __init__(self, embedding_size):
        super().__init__()
        self.embedding_size = embedding_size
        self.hi = nn.Linear(embedding_size, embedding_size, bias=False)
        self.ho = nn.Linear(embedding_size, embedding_size, bias=False)
        self.hv = nn.Linear(embedding_size, 1, bias=False)

    def project_keys(self, keys):
        return self.hi(keys)

    def calc_attention(self, projected_keys, queries):
        queries = self.ho(queries)
        x = torch.tanh(projected_keys + queries)
        x = self.hv(x).squeeze(-1)
        x = F.softmax(x, dim=0)
        return x

    def apply_attention(self, att, values):
        M, N, E = values.size()
        values = Hadamard(values, att.unsqueeze(-1).expand(M, N, E))
        context = values.sum(dim=0)
        return context
