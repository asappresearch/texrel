"""
adapted from https://github.com/jacobandreas/tre/blob/master/evals2.py
forked 2021, feb 7
the original code was provided under apache 2 license
"""
from typing import List, Dict, Tuple, Any, Union, Callable, cast
import abc

import numpy as np
import torch
from torch import nn
from torch import optim


def flatten(not_flat: Union[Tuple[Any, Any], Any]) -> Tuple[Any, ...]:
    """
    Given an input which is a tree of tuples,
    returns a single tuple containing the leaves
    """
    if not isinstance(not_flat, tuple):
        return (not_flat,)

    out: Tuple[Any, ...] = ()
    for ll in not_flat:
        out = out + flatten(ll)
    return out


class ProjectionSumCompose(nn.Module):
    """
    following section 7 of "Measuring Compositionality" paper,
    composes incoming tensors by projecting each, and then summing.
    These tensors will in practice
    each be the result of embedding oracle structure trees
    or one of the oracle structure tree leaves

    Parameters
    ----------
    num_terms: int
        this composition will always compose a fixed number of incoming tensors
        num_terms is the number of these incoming tensors
    vocab_size: int
        used to reshape incoming tensors into [vocab_size + 2][message_length]
    msg_len: int
        used to create the projection matrices
    bias: bool
        if True, then we add a bias term to the projection operation (note that
        the original paper's implicitly sets this to True, but the formula
        in the paper corresponds to bias=False)
    """
    def __init__(self, num_terms: int, vocab_size: int, msg_len: int, bias: bool):
        super().__init__()
        self.vocab_size = vocab_size
        self.msg_len = msg_len
        self.num_terms = num_terms
        self.proj_l = nn.ModuleList()
        for i in range(num_terms):
            self.proj_l.append(nn.Linear(msg_len, msg_len, bias=bias))

    def forward(self, *args: torch.Tensor) -> torch.Tensor:
        """
        composes incoming embedded oracle structures by projecting them, then summing.

        The incoming tensors are assumed to be flattened vector representations
        of utterances. we will unflatten them into per-token distributions
        over vocabulary, and then project the resulting two-dimensional tensors
        (we unsqueeze the first dimension over batch_size=1 since Linear expects
        batch size first dimension)

        Parameters
        ----------
        *args: torch.Tensor
            tensors to compose, as float tensors

        Returns
        -------
        torch.Tensor
            sum of projections of incoming tensors
        """
        x_l = [arg.view(1, self.vocab_size, self.msg_len) for arg in args]
        x_l = [proj(x) for x, proj in zip(x_l, self.proj_l)]
        x = torch.stack(x_l)
        x = x.sum(dim=0)
        return x.view(1, (self.vocab_size) * self.msg_len)


class Distance(nn.Module, abc.ABC):
    def __call__(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return super().__call__(pred, target)

    @abc.abstractmethod
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pass


class L1Dist(Distance):
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return torch.abs(pred - target).sum()


class CosDist(Distance):
    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        nx, ny = nn.functional.normalize(x), nn.functional.normalize(y)
        return 1 - (nx * ny).sum()


class TREModel(nn.Module):
    def __init__(
            self,
            oracle_vocab_size: int,
            repr_size: int,
            comp_fn: Callable,
            distance_fn: Distance,
            zero_init: bool):
        """
        Parameters
        ----------
        oracle_vocab_size: int
            size of vocabulary in the oracle_structure trees passed to forward
        repr_size: int
            size of the representation passed into forward. This will also be used
            as the dimension of the Embedding for embedding the oracle structure
        comp_fn: Callable
            used to compose two embedded oracle structure nodes
        distance_fn: Callable
            used to obtain distance between passed-in rep, and embedded oracle
            struture
        zero_init: bool
            if True, then we set the initial oracle Embedding weights to zero
        """
        super().__init__()
        self.embedding = nn.Embedding(oracle_vocab_size, repr_size)
        if zero_init:
            self.embedding.weight.data.zero_()
        self.comp_fn = comp_fn
        self.distance_fn = distance_fn

    def embed_oracle(self, oracle_structure: Union[torch.Tensor, Tuple[Any, Any]]) -> torch.Tensor:
        """
        Parameters
        ----------
        oracle_structure Union[torch.Tensor, Tuple['oracle_structure', 'oracle_structure']]
            tree of long tensors, where each leaf is a long tensor containing a single int
            this is a representation of eg '(left-of((brown, box), (red, square))'

        Returns
        -------
        embedding of the oracle structure, by using self.emb on the leaves,
            and recursively using self.comp_fn on parent nodes
        """
        if isinstance(oracle_structure, tuple):
            composed_sub_structs = tuple(self.embed_oracle(sub_struct) for sub_struct in oracle_structure)
            return self.comp_fn(*composed_sub_structs)
        return self.embedding(oracle_structure)

    def forward(self, rep: torch.Tensor, oracle_structure: Union[
            torch.Tensor, Tuple[Any, Any]]):
        """
        Parameters
        ----------
        rep: torch.Tensor
            representation of a single input. This could be an utterance for example
        oracle_structure: Union[torch.Tensor, Tuple['oracle_structure', 'oracle_structure']]
            oracle structure, as a tree of LongTensors. each LongTensor should contain a single integer
        """
        return self.distance_fn(self.embed_oracle(oracle_structure), rep)


def evaluate_(
        reps: torch.Tensor,
        oracle_structures: List[Union[str, Tuple[Any, Any]]],
        comp_fn: Callable,
        distance_fn: Distance,
        tre_lr: float,
        quiet: bool,
        steps: int,
        include_pred: bool,
        max_samples: int,
        zero_init: bool) -> Union[
            float,
            Tuple[List[float], Dict[str, torch.Tensor], List[torch.Tensor]]
        ]:
    """
    Parameters
    ----------
    reps: torch.Tensor
        representations we wish to fit embedded oracle_structures to
        reps should be a 2-dimensional tensor of size [num_exmaples, num_features]
        if reps represent uttearnces, each token should be one-hotted, and
        then the one-hotted token representations should be concatenated
        so that reps would be of dimension [N][S * V], where
        [S] is sequence length, [N] is mini-batch size, and [V]
        is vocab size
    oracle_structures: List[Union[str, Tuple['oracle_structure', 'oracle_structure']]]
        oracle_structures is a list of oracle structures, where each oracle structure
        is a tree with tuples as the non-leaf nodes, and words as the leaf nodes, eg
        '(left-of((brown, box), (red, square))'
    comp_fn: Callable
        takes in a tuple of embedded oracle structures, and returns a single tensor
        composing the tuple of embedded oracle structures
    max_samples: int
        how many samples to use (more gives more reproducibility; fewer runs faster)
    distance_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor]
        function which takes in two representations, and returns a scalar tensor
        representing some kind of distance between the two representations
    """
    if len(oracle_structures) > max_samples:
        idxes = np.random.choice(len(oracle_structures), max_samples, replace=False)
        oracle_structures = [oracle_structures[idx] for idx in idxes]
        reps = reps[idxes]

    oracle_vocab: Dict[str, int] = {}
    for structure in oracle_structures:
        toks = flatten(structure)
        for tok in toks:
            if tok not in oracle_vocab:
                oracle_vocab[tok] = len(oracle_vocab)

    def words_tree_to_idxes_tree(oracle_structure: Union[str, Tuple[Any, Any]]):
        if isinstance(oracle_structure, tuple):
            return tuple(words_tree_to_idxes_tree(sub_struct) for sub_struct in oracle_structure)
        return torch.LongTensor([oracle_vocab[oracle_structure]])

    assert reps.dtype == torch.float32
    N, repr_size = reps.size()

    reps = reps.cpu()
    reps_split = reps.split(dim=0, split_size=1)
    tre_model = TREModel(
        oracle_vocab_size=len(oracle_vocab),
        repr_size=repr_size,
        comp_fn=comp_fn,
        distance_fn=distance_fn,
        zero_init=zero_init)
    opt = optim.Adam(tre_model.parameters(), lr=tre_lr)

    oracle_structures_idxes = [
        words_tree_to_idxes_tree(oracle_structure)for oracle_structure in oracle_structures]

    for t in range(steps):
        opt.zero_grad()
        errs = [tre_model(
            rep=rep, oracle_structure=oracle_structure) for rep, oracle_structure in
            zip(reps_split, oracle_structures_idxes)]
        loss = torch.stack(errs).sum()
        loss.backward()
        if not quiet and t % 100 == 0:
            print(' %.3f' % loss.item())
        opt.step()

    final_errs = [err.item() for err in errs]
    if include_pred:
        lexicon = {
            word: tre_model.embedding(torch.LongTensor([word_idx])).data.cpu().numpy()
            for word, word_idx in oracle_vocab.items()
        }
        composed = [
            tre_model.embed_oracle(oracle_structure_idxes) for oracle_structure_idxes in oracle_structures_idxes]
        return final_errs, lexicon, composed
    else:
        return sum(final_errs) / len(final_errs)


def evaluate(
        reps: torch.Tensor,
        oracle_structures: List[Union[str, Tuple[Any, Any]]],
        comp_fn: Callable,
        distance_fn: Distance,
        tre_lr: float,
        quiet: bool,
        steps: int,
        max_samples: int,
        zero_init: bool) -> float:
    return cast(float, evaluate_(
        reps=reps, oracle_structures=oracle_structures, comp_fn=comp_fn,
        distance_fn=distance_fn, quiet=quiet, steps=steps, zero_init=zero_init,
        max_samples=max_samples,
        include_pred=False, tre_lr=tre_lr))
