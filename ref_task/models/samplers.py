"""
Various ways of handling decision points, such as stochastic etc

Each of the child classes is going to store state internally, so we can later calculate loss
easily, without needing to handle each case specifically in the main other code. differences
should be encapsulated here, black-box ish
"""
from abc import ABC, abstractmethod

import torch
from torch import nn
import torch.nn.functional as F

from ulfs.stochastic_trajectory import StochasticTrajectory
from ulfs import rl_common


class Sampler(ABC, nn.Module):
    @abstractmethod
    def forward(self, utt_logits):
        """
        Input are logits. NOT softmaxed.

        returns probs

        This will return a probability distribution over token types.
        this is true even for REINFORCE
        any stochastic trajectory will be a property of the object
        reinforce loss is calcualted yb calling calc_loss
        """
        pass

    @abstractmethod
    def calc_loss(self, rewards):
        """
        this method should in addition to calculating the loss, also
        implicitly clear any state that we need to clear prior to next
        forward pass

        this should *only* return a non-zero loss if backprop from the criterion
        at the end of the network wont already backprop through this

        eg:
        - REINFORCE => need to implement this, since backprop wont go further than
          our stochastic sampler
        - Gumbel, softmax => no need to implement this, since normal backprop will
          go through the sampler anyway
        """
        pass


class SoftmaxSampler(Sampler):
    """
    subsumes the case where no noise by implicitly setting the noise to zero
    but noise might be good sometimes too :)
    """
    def __init__(self, sampler_gaussian_noise):
        super().__init__()
        self.gaussian_noise = sampler_gaussian_noise

    def forward(self, utt_logits):
        """
        assumed that the probs are over the last dim
        """
        x = F.softmax(utt_logits, dim=-1)
        if self.gaussian_noise > 0 and self.training:
            shape = list(x.size())
            x = x + self.gaussian_noise * torch.randn(*shape, device=x.device)
        return x

    def calc_loss(self, rewards):
        """
        no additional loss to calculate, since will backprop through naturally,
        so we just reutrn 0

        no internal state to clear either, so nothing to do
        """
        return 0


class GumbelSampler(Sampler):
    def __init__(self, sampler_tau, sampler_hard):
        super().__init__()
        self.tau = sampler_tau
        self.hard = sampler_hard

    def forward(self, x):
        """
        assumes that probability distribution is over final dimension
        """
        if self.training:
            shape_all = list(x.size())
            x = x.contiguous().view(-1, shape_all[-1])
            log_x = x
            x = F.gumbel_softmax(log_x, tau=self.tau, hard=self.hard)
            x = x.view(*shape_all)
        else:
            # not sure if this is the best way of doing this...
            x = F.softmax(x, dim=-1)
            _, idxes = x.max(dim=-1)

            onehot_t = torch.zeros_like(x, device=x.device)
            onehot_t.scatter_(-1, idxes.unsqueeze(-1), 1.0)
            x = onehot_t

        return x

    def calc_loss(self, rewards):
        """
        no additional loss to calculate, since will backprop through naturally,
        so we just reutrn 0
        """
        return 0


class REINFORCESampler(Sampler):
    def __init__(self, sampler_ent_reg, sampler_baseline_lambda):
        super().__init__()
        self.st = StochasticTrajectory()
        self.baseline = 0
        self.ent_reg = sampler_ent_reg
        self.baseline_lambda = sampler_baseline_lambda

    def forward(self, x, batch_idxes=None):
        """
        x is assumed to be [M][N][V]

        not sure how to handle batch_idxes for now. think about it when we need them....
        """
        incoming_shape = list(x.size())
        probs = F.softmax(x, dim=-1)
        s = rl_common.draw_categorical_sample(
            action_probs=probs,
            batch_idxes=batch_idxes
        )
        self.st.append_stochastic_sample(s)

        onehot_t = torch.zeros(*incoming_shape, device=x.device)
        onehot_t.scatter_(-1, s.actions.unsqueeze(-1), 1.0)

        return onehot_t

    def calc_loss(self, rewards):
        rewards = rewards.detach()
        baselined_rewards = rewards - self.baseline
        rl_loss = self.st.calc_loss(rewards=baselined_rewards)
        self.baseline = self.baseline_lambda * self.baseline + (1 - self.baseline_lambda) * rewards.mean().item()
        ent_loss = - self.ent_reg * self.st.entropy

        # clear state
        self.st = StochasticTrajectory()

        return rl_loss + ent_loss


def build_sampler(sampler_model, **kwargs):
    SamplerClass = globals()[f'{sampler_model}Sampler']
    child_kwargs = {}
    if sampler_model == 'Gumbel':
        child_kwargs = {
            'sampler_tau': kwargs['sampler_tau'],
            'sampler_hard': kwargs['sampler_hard'],
        }
    elif sampler_model == 'Softmax':
        child_kwargs = {
            'sampler_gaussian_noise': kwargs['sampler_gaussian_noise'],
        }
    elif sampler_model == 'REINFORCE':
        child_kwargs = {
            'sampler_ent_reg': kwargs['sampler_ent_reg'],
            'sampler_baseline_lambda': kwargs['sampler_baseline_lambda'],
        }
    return SamplerClass(**child_kwargs)
