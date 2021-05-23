"""
take hypothesis and embed it, eg using rnn encoder
"""
import abc
import torch
from torch import nn

from sru import SRU


class LinguisticEncoder(nn.Module, abc.ABC):
    def __call__(self, utts: torch.Tensor) -> torch.Tensor:
        return super().__call__(utts=utts)

    @abc.abstractmethod
    def forward(self, utts: torch.Tensor) -> torch.Tensor:
        pass


class FCModel(LinguisticEncoder):
    """
    relations in forward() are one-hotted, though could be softened too
    (so no embedding layer, just linear)
    """
    def __init__(self, vocab_size, in_seq_len, embedding_size, dropout, num_layers):
        super().__init__()
        self.embedding_size = embedding_size
        self.num_layers = num_layers
        self.vocab_size = vocab_size
        self.in_seq_len = in_seq_len

        self.h1 = nn.Linear(in_seq_len * vocab_size, embedding_size)

        self.layers = nn.ModuleList()
        for i in range(num_layers - 1):
            self.layers.append(nn.Dropout(dropout))
            self.layers.append(nn.Tanh())
            self.layers.append(nn.Linear(embedding_size, embedding_size))

    def forward(self, utts: torch.Tensor) -> torch.Tensor:
        """
        incoming relations is [M][N][V]
        ([V] is vocab_size, [M] is seq_len, [N] is batch_size)
        """
        M, N, V = utts.size()
        assert M == self.in_seq_len
        assert V == self.vocab_size
        utts = self.h1(utts.transpose(0, 1).contiguous().view(N, M * V))
        for i, l in enumerate(self.layers):
            utts = l(utts)
        return utts


class RNNModel(LinguisticEncoder):
    """
    Just use a standard RNN to embed the relation as if it is an utterance (which
    it often is...))
    """
    def __init__(self, vocab_size, in_seq_len, embedding_size, dropout, num_layers, rnn_type):
        super().__init__()
        self.embedding_size = embedding_size
        self.num_layers = num_layers
        self.vocab_size = vocab_size
        self.in_seq_len = in_seq_len
        self.rnn_type = rnn_type

        self.v2e = nn.Linear(vocab_size, embedding_size)
        if rnn_type != 'SRU':
            RNN = getattr(nn, rnn_type)
        else:
            RNN = SRU
        rnn_constructor_args = {}
        if rnn_type == 'SRU':
            rnn_constructor_args['nn_rnn_compatible_return'] = True
        self.rnn = RNN(embedding_size, embedding_size, **rnn_constructor_args)

    def forward(self, utts: torch.Tensor) -> torch.Tensor:
        """
        incoming relations is [M][N][V]
        ([V] is vocab_size, [M] is seq_len, [N] is batch_size)
        """
        M, N, V = utts.size()
        assert M == self.in_seq_len
        assert V == self.vocab_size
        embeddings = self.v2e(utts)
        self.rnn.flatten_parameters()
        output, state = self.rnn(embeddings)
        state = state[-1]
        return state


def build_linguistic_encoder(params, utt_len: int, vocab_size: int) -> LinguisticEncoder:
    p = params
    LinguisticEncoderClass = globals()[f'{p.linguistic_encoder}Model']
    params = {
        'vocab_size': vocab_size,
        'in_seq_len': utt_len,
        'embedding_size': p.embedding_size,
        'dropout': p.dropout,
        'num_layers': p.linguistic_encoder_num_layers,
    }
    if p.linguistic_encoder in ['RNN']:
        params['rnn_type'] = p.linguistic_encoder_rnn_type
    linguistic_encoder: LinguisticEncoder = LinguisticEncoderClass(**params)
    return linguistic_encoder
