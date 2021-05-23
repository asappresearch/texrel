"""
note to self:
decoder: embedding => language
encoder: language => embedding
"""
import torch
from torch import nn
from torch.autograd import Variable
import torch.nn.functional as F

from ulfs import alive_sieve, gumbel


class DiscreteEncoder(nn.Module):
    """
    Encoder is non-stochastic, fully differentiable

    Takes language => embedding

    Input is vocab indexes, in LongTensor

    input should NOT be a Variable, should just be a normal tensor
    """
    def __init__(self, vocab_size, embedding_size):
        super().__init__()
        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        self.d2e = nn.Embedding(vocab_size, embedding_size)
        self.e2d = nn.Linear(embedding_size, vocab_size)
        # share weights between the embedding and its inverse
        # print('self.e2d.weight.size()', self.e2d.weight.size(), 'enoder.d2e.weight.size()', self.d2e.weight.size())
        # self.e2d.weight.data = self.d2e.weight.data
        self.rnn = nn.GRUCell(embedding_size, embedding_size)

    def forward(self, utterance):
        """
        input: letters
        output: embedding
        """
        batch_size = utterance.size()[0]
        utterance_max = utterance.size()[1]
        sieve = alive_sieve.AliveSieve(batch_size=batch_size, enable_cuda=False)
        state = torch.FloatTensor(batch_size, self.embedding_size).fill_(0)
        output_state = state.clone()
        for t in range(utterance_max):
            emb = self.d2e(utterance[:, t])
            state = self.rnn(emb, state)
            output_state[sieve.global_idxes] = state
            sieve.mark_dead(utterance[:, t] == 0)
            if sieve.all_dead():
                break

            utterance = utterance[sieve.alive_idxes]
            state = state[sieve.alive_idxes]
            sieve.self_sieve_()
        return output_state


class OnehotEncoder(nn.Module):
    """
    Encoder is non-stochastic, fully differentiable

    Takes language => embedding

    Input is one-hot, in FloatTensor
    """
    def __init__(self, vocab_size, embedding_size, no_terminator):
        super().__init__()
        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        self.no_terminator = no_terminator
        self.d2e = nn.Linear(vocab_size, embedding_size)
        # share weights between the embedding and its inverse
        # self.e2d.weight.data = self.d2e.weight.data
        self.rnn = nn.GRUCell(embedding_size, embedding_size)
        self.e2d = nn.Linear(embedding_size, vocab_size)

    def forward(self, utterance):
        """
        input: letters
        output: embedding
        """
        batch_size = utterance.size()[0]
        utterance_max = utterance.size()[1]
        sieve = alive_sieve.AliveSieve(batch_size=batch_size, enable_cuda=False)
        state = torch.FloatTensor(batch_size, self.embedding_size).fill_(0)
        output_state = state.clone()
        for t in range(utterance_max):
            emb = self.d2e(utterance[:, t])
            state = self.rnn(emb, state)
            output_state[sieve.global_idxes] = state
            if not self.no_terminator:
                sieve.mark_dead(utterance.data[:, t, 0] == 1)
                if sieve.all_dead():
                    break

            utterance = utterance[sieve.alive_idxes]
            state = state[sieve.alive_idxes]
            sieve.self_sieve_()
        return output_state


class OnehotCNNEncoder(nn.Module):
    """
    should always be no_terminator=True
    """
    def __init__(self, vocab_size, embedding_size, no_terminator=False):
        super().__init__()
        assert no_terminator
        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        self.no_terminator = no_terminator

        # i dunno, 2, 3 layers?
        # we have the inputs are one-hot over vocab_size, so [6][3] or so
        # treat the vocab_size as channels, so 6 channels, width 6
        # throw on a 16 channel, kernel 3, pad 1
        # then a 32 channel, kernel 3, pad 1

        # channels = vocab_size
        # print('vocab_size', vocab_size)
        self.conv1 = nn.Conv1d(in_channels=vocab_size, out_channels=8, kernel_size=3, padding=0)
        self.conv2 = nn.Conv1d(in_channels=8, out_channels=16, kernel_size=3, padding=0)
        # self.conv3 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3, padding=0)
        # self.conv4 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, padding=0)
        self.mp = nn.AdaptiveMaxPool1d(output_size=1)

        # self.d2e = nn.Linear(vocab_size, embedding_size)
        # self.rnn = nn.GRUCell(embedding_size, embedding_size)
        # self.e2d = nn.Linear(embedding_size, vocab_size)

    def forward(self, utterance):
        """
        input: letters
        output: embedding
        """
        # print('utterance.size()', utterance.size())
        batch_size = utterance.size()[0]
        utterance = utterance.transpose(1, 2)
        # print('utterance.size()', utterance.size())

        x = F.relu(self.conv1(utterance))
        x = F.relu(self.conv2(x))
        # x = F.relu(self.conv3(x))
        # x = F.relu(self.conv4(x))
        # print('x.size()', x.size())
        x = self.mp(x)
        x = x.view(batch_size, -1)
        return x


class StochasticDecoder(nn.Module):
    def __init__(self, encoder, utterance_max):
        super().__init__()
        self.utterance_max = utterance_max
        self.vocab_size = encoder.vocab_size
        self.embedding_size = encoder.embedding_size
        self.e2d = encoder.e2d
        self.d2e = encoder.d2e
        self.rnn = encoder.rnn

    def forward(self, x, global_idxes):
        """
        Input is a batch of embedding vectors
        We'll convert to discrete letters
        """
        batch_size = x.size()[0]
        state = x
        global_idxes = global_idxes.clone()
        # note that this sieve might start off smaller than the global batch_size
        sieve = alive_sieve.AliveSieve(batch_size=batch_size, enable_cuda=False)
        last_token = torch.LongTensor(batch_size).fill_(0)
        utterance = torch.LongTensor(batch_size, self.utterance_max).fill_(0)
        N_outer = torch.LongTensor(batch_size).fill_(self.utterance_max)
        node_idxes_pairs = []
        entropy = 0
        for t in range(self.utterance_max):
            emb = self.d2e(Variable(last_token))
            state = self.rnn(emb, state)
            token_logits = self.e2d(state)
            token_probs = F.softmax(token_logits)
            token_node = torch.multinomial(token_probs)
            node_idxes_pairs.append({
                'node': token_node,
                'global_idxes': global_idxes[sieve.global_idxes]
            })
            token_probs = token_probs + 1e-8
            entropy -= (token_probs * token_probs.log()).sum()
            token = token_node.data.view(-1)
            utterance[:, t][sieve.global_idxes] = token
            last_token = token
            sieve.mark_dead(last_token == 0)
            sieve.set_global_dead(N_outer, t + 1)
            if sieve.all_dead():
                break
            state = state[sieve.alive_idxes]
            last_token = last_token[sieve.alive_idxes]
            sieve.self_sieve_()
        return node_idxes_pairs, utterance, N_outer, entropy


class StochasticDecoderEncoder(nn.Module):
    def __init__(self, vocab_size, embedding_size, utterance_max):
        super().__init__()
        self.encoder = DiscreteEncoder(vocab_size=vocab_size, embedding_size=embedding_size)
        self.decoder = StochasticDecoder(self.encoder, utterance_max=utterance_max)

    def forward(self, x, global_idxes):
        """
        input: embedding
        output: same embedding, reoncstructed, via discrete letters
        """
        # print('global_idxes', global_idxes)
        node_idxes_pairs, utterance, N_outer, entropy = self.decoder(x, global_idxes)
        reconst = self.encoder(utterance)
        return node_idxes_pairs, utterance, N_outer, entropy, reconst


class DifferentiableDecoder(nn.Module):
    """
    embedding => language

    Output is one-hot type distributions over lettrs, rather than
    hard argmaxd letter indices
    Therefore differentiable

    should we gumbelize it? I'm not sure :(
    """
    def __init__(self, encoder):
        super().__init__()
        self.vocab_size = encoder.vocab_size
        self.embedding_size = encoder.embedding_size
        self.e2d = nn.Linear(self.embedding_size, self.vocab_size)
        self.d2e = nn.Linear(self.vocab_size, self.embedding_size)
        self.rnn = nn.GRUCell(self.embedding_size, self.embedding_size)

    def forward(self, x, max_len):
        """
        Input is a batch of embedding vectors
        We'll convert to discrete letters
        """
        batch_size = x.size()[0]
        state = x
        last_token = Variable(torch.FloatTensor(batch_size, self.vocab_size).fill_(0))
        utterance = Variable(torch.FloatTensor(batch_size, max_len, self.vocab_size).fill_(0))
        for t in range(max_len):
            emb = self.d2e(last_token)
            state = self.rnn(emb, state)
            token_logits = self.e2d(state)
            token_probs = F.softmax(token_logits)
            utterance[:, t] = token_probs
            last_token = token_probs
        return utterance


class DifferentiableEncoderDecoder(nn.Module):
    """
    input: discrete letters
    output: reconstructed letters, in one hot type format (prob dist)
    """
    def __init__(self, vocab_size, embedding_size):
        super().__init__()
        self.encoder = OnehotEncoder(vocab_size=vocab_size, embedding_size=embedding_size)
        self.decoder = DifferentiableDecoder(encoder=self.encoder)

    def forward(self, x):
        """
        input: discrete letters [batch_size][seq_len]
        output: one-hot type distrib over letters, for the length of the word  [batch_size][seq_len][vocab_size]
        """
        embedding = self.encoder(x)
        print('embedding', embedding)
        max_len = x.size()[1]
        x_reconst_onehottish = self.decoder(embedding, max_len=max_len)
        return embedding, x_reconst_onehottish


class GumbelDecoder(nn.Module):
    """
    embedding => language

    Output is one-hot type distributions over lettrs, rather than
    hard argmaxd letter indices
    Differentiable
    """
    def __init__(self, vocab_size, embedding_size, utterance_max, tau, no_terminator=False):
        super().__init__()
        self.utterance_max = utterance_max
        self.no_terminator = no_terminator
        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        self.tau = tau
        # note to self: share weights?
        self.d2e = nn.Linear(self.vocab_size, self.embedding_size)
        self.rnn = nn.GRUCell(self.embedding_size, self.embedding_size)
        self.e2d = nn.Linear(self.embedding_size, self.vocab_size)

    def forward(self, x):
        """
        Input is a batch of embedding vectors
        We'll convert to discrete one-hot letters
        """
        batch_size = x.size()[0]
        state = x
        last_token = Variable(torch.FloatTensor(batch_size, self.vocab_size).fill_(0))
        N = torch.LongTensor(batch_size).fill_(self.utterance_max)
        alive_mask = torch.ByteTensor(batch_size).fill_(1)
        dead_idxes = []
        utterance = Variable(torch.FloatTensor(batch_size, self.utterance_max, self.vocab_size).fill_(0))
        for t in range(self.utterance_max):
            emb = self.d2e(last_token)
            state = self.rnn(emb, state)
            token_logits = self.e2d(state)
            token_draw = gumbel.gumbel_softmax(token_logits, tau=self.tau, hard=True)
            _, tok_max = token_draw.data.max(1)
            if not self.no_terminator:
                ended_mask = token_draw.data[:, 0] == 1
                ended_idxes = ended_mask.nonzero().long().view(-1)
                if len(ended_idxes) > 0:
                    alive_mask[ended_idxes] = 0
                    dead_idxes = (1 - alive_mask).nonzero().long().view(-1)
                    N[ended_idxes] = torch.min(N[ended_idxes], torch.LongTensor([t]))
                if len(dead_idxes) > 0:
                    token_draw[dead_idxes] = 0
            utterance[:, t] = token_draw
            last_token = token_draw
        _, utt_max = utterance.max(2)
        return utterance, N


class GumbelDecoderEncoder(nn.Module):
    def __init__(self, vocab_size, embedding_size, utterance_max, tau, no_terminator=False):
        super().__init__()
        self.embedding_size = embedding_size
        self.encoder = OnehotEncoder(vocab_size=vocab_size, embedding_size=embedding_size, no_terminator=no_terminator)
        self.decoder = GumbelDecoder(self.encoder, utterance_max=utterance_max, tau=tau, no_terminator=no_terminator)

    def forward(self, x, global_idxes):
        """
        input: embedding
        output: same embedding, reoncstructed, via discrete letters
        """
        utterance, N = self.decoder(x)
        reconst = self.encoder(utterance)
        return utterance, N, reconst


class GumbelDecoderCNNEncoder(nn.Module):
    def __init__(self, vocab_size, embedding_size, utterance_max, tau, no_terminator=False):
        super().__init__()
        assert no_terminator
        self.embedding_size = embedding_size
        self.encoder = OnehotCNNEncoder(
            vocab_size=vocab_size, embedding_size=embedding_size, no_terminator=no_terminator)
        self.decoder = GumbelDecoder(self.encoder, utterance_max=utterance_max, tau=tau, no_terminator=no_terminator)
        self.h1 = nn.Linear(16, embedding_size)

    def forward(self, x, global_idxes):
        """
        input: embedding
        output: same embedding, reoncstructed, via discrete letters
        """
        utterance, N = self.decoder(x)
        reconst = self.encoder(utterance)
        reconst = F.tanh(reconst)
        reconst = self.h1(reconst)
        return utterance, N, reconst
