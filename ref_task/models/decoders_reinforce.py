"""
stochastic versions of the decoders in decoders_differentiable

these decoders:
- produce discrete outputs
- work in / need REINFORCE settings
"""
import abc
from typing import Tuple, Optional

import torch
from torch import nn
import torch.nn.functional as F

from ulfs import rl_common
from ulfs.stochastic_trajectory import StochasticTrajectory


class StochasticDecoder(nn.Module, abc.ABC):
    def __call__(self, embedding: torch.Tensor) -> Tuple[torch.Tensor, Optional[StochasticTrajectory]]:
        return super().__call__(embedding=embedding)

    @abc.abstractmethod
    def forward(self, embedding: torch.Tensor) -> Tuple[torch.Tensor, Optional[StochasticTrajectory]]:
        pass


class StochasticClassifier(StochasticDecoder):
    """
    so, if we have a standard multi-class output, that we could put
    through CrossEntropyLoss, if we feed it through this, we do
    the stochastic thing

    But, this handles doing argmax instead during eval
    """
    def __init__(self):
        super().__init__()

    def forward(self, embedding: torch.Tensor) -> Tuple[torch.Tensor, Optional[StochasticTrajectory]]:
        """
        embedding should be [N][C] where [C] is number of classes
        """
        assert len(embedding.size()) == 2
        N, C = embedding.size()
        device = embedding.device

        if self.training:
            probs = F.softmax(embedding, dim=-1)
            batch_idxes = torch.ones(N, device=device, dtype=torch.int64).cumsum(-1) - 1
            s = rl_common.draw_categorical_sample(
                action_probs=probs, batch_idxes=batch_idxes)
            tokens = s.actions
        else:
            _, tokens = embedding.max(dim=-1)
            s = None
        return tokens, s


class FlatDecoder(StochasticDecoder):
    def __init__(self, embedding_size, out_seq_len, vocab_size):
        super().__init__()
        self.out_seq_len = out_seq_len
        self.vocab_size = vocab_size

        self.relation_head = nn.Linear(embedding_size, vocab_size * out_seq_len)

    def forward(self, embedding: torch.Tensor) -> Tuple[torch.Tensor, Optional[StochasticTrajectory]]:
        """
        embedding should be [N][E]
        """
        assert len(embedding.size()) == 2
        batch_size = embedding.size(0)
        device = embedding.device

        relations = self.relation_head(embedding)
        relations = relations.view(batch_size, self.out_seq_len, self.vocab_size)
        relations = relations.transpose(0, 1)

        if self.training:
            probs = F.softmax(relations, dim=-1)
            batch_idxes = torch.ones(batch_size, device=device, dtype=torch.int64).cumsum(-1) - 1
            s = rl_common.draw_categorical_sample(
                action_probs=probs, batch_idxes=batch_idxes)
            utterances = s.actions
        else:
            _, utterances = relations.max(dim=-1)
            s = None
        return utterances, s


class RNNDecoder(StochasticDecoder):
    def __init__(
            self, embedding_size, out_seq_len, vocab_size):
        super().__init__()
        # self.input_size = input_size
        self.embedding_size = embedding_size
        self.out_seq_len = out_seq_len
        self.vocab_size = vocab_size

        # d2e "discrete to embed"
        self.d2e = nn.Embedding(vocab_size, embedding_size)
        self.rnn = nn.GRUCell(embedding_size, embedding_size)
        self.e2d = nn.Linear(embedding_size, vocab_size)

    def forward(self, embedding: torch.Tensor) -> Tuple[torch.Tensor, Optional[StochasticTrajectory]]:
        """
        x is [N][E]  where [N] is batch size, and [E] is input size
        """
        batch_size = embedding.size(0)
        device = embedding.device

        state = embedding

        global_idxes = torch.ones(batch_size, dtype=torch.int64, device=device).cumsum(dim=-1) - 1
        last_token = torch.zeros(batch_size, dtype=torch.int64, device=device)
        utterance = torch.zeros(self.out_seq_len, batch_size, dtype=torch.int64, device=device)

        if self.training:
            stochastic_trajectory = StochasticTrajectory()

        for t in range(self.out_seq_len):
            emb = self.d2e(last_token)
            state = self.rnn(emb, state)
            token_logits = self.e2d(state)
            token_probs = F.softmax(token_logits, dim=-1)

            if self.training:
                s = rl_common.draw_categorical_sample(
                    action_probs=token_probs, batch_idxes=global_idxes)
                stochastic_trajectory.append_stochastic_sample(s=s)
                token = s.actions.view(-1)
            else:
                _, token = token_probs.max(-1)
            utterance[t] = token
            last_token = token

        if self.training:
            return utterance, stochastic_trajectory
        else:
            return utterance, None
