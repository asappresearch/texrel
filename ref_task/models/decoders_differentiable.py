"""
differentiable, as the module name says, cf decoders_reinforce.py, which
needs REINFORCE to learn

returns logits
"""
import abc

import torch
from torch import nn
import torch.nn.functional as F


class DifferentiableDecoder(nn.Module, abc.ABC):
    def __call__(self, embedding: torch.Tensor) -> torch.Tensor:
        return super().__call__(embedding=embedding)

    @abc.abstractmethod
    def forward(self, embedding: torch.Tensor) -> torch.Tensor:
        pass


class IdentityDecoder(DifferentiableDecoder):
    """
    We'll use this with eg UniversalTransformer embedding, which outputs
    embedding as a sequence anyway, so we just need to argmax
    """
    def __init__(self):
        super().__init__()

    def forward(self, embedding: torch.Tensor) -> torch.Tensor:
        """
        embedding should be [M][N][V]
        """
        assert len(embedding.size()) == 3
        M, N, V = embedding.size()

        return embedding


class FlatDecoder(DifferentiableDecoder):
    def __init__(self, embedding_size, out_seq_len, vocab_size):
        super().__init__()
        self.out_seq_len = out_seq_len
        self.vocab_size = vocab_size

        self.relation_head = nn.Linear(embedding_size, vocab_size * out_seq_len)

    def forward(self, embedding: torch.Tensor) -> torch.Tensor:
        """
        embedding should be [N][E]
        """
        assert len(embedding.size()) == 2
        batch_size = embedding.size(0)

        relations = self.relation_head(embedding)
        relations = relations.view(batch_size, self.out_seq_len, self.vocab_size)
        relations = relations.transpose(0, 1)

        return relations


class RNNDecoder(DifferentiableDecoder):
    """
    TODO: Gumbel-enable this
    """
    def __init__(self, embedding_size, out_seq_len, vocab_size, rnn_type='gru'):
        super().__init__()
        self.out_seq_len = out_seq_len
        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        self.rnn_type = rnn_type

        self.e2d = nn.Linear(self.embedding_size, self.vocab_size)
        self.d2e = nn.Linear(self.vocab_size, self.embedding_size)
        if rnn_type == 'gru':
            self.rnn = nn.GRUCell(self.embedding_size, self.embedding_size)
        elif rnn_type == 'lstm':
            self.rnn = nn.LSTMCell(self.embedding_size, self.embedding_size)
        else:
            raise Exception('rnn type %s not known' % rnn_type)

    def forward(self, embedding: torch.Tensor) -> torch.Tensor:
        """
        embeddings should be [N][E] (since not a sequence yet)
        """
        assert len(embedding.size()) == 2
        batch_size = embedding.size()[0]
        device = embedding.device

        state = embedding
        cell = torch.zeros(batch_size, self.embedding_size, device=device)
        last_token = torch.zeros(batch_size, self.vocab_size, device=device)
        utterance = torch.zeros(self.out_seq_len, batch_size, self.vocab_size, device=device)
        for t in range(self.out_seq_len):
            emb = self.d2e(last_token)
            if self.rnn_type == 'lstm':
                state, cell = self.rnn(emb, (state, cell))
            else:
                state = self.rnn(emb, state)
            token_logits = self.e2d(state)
            token_probs = F.softmax(token_logits, dim=-1)
            utterance[t] = token_logits
            last_token = token_probs
        return utterance
