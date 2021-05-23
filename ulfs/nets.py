import torch
from torch import nn
import torch.nn.functional as F
from torch.distributions.categorical import Categorical

from . import language_model
from . import layer_utils


class StateCNN(nn.Module):
    def __init__(self, net_string, in_width, in_height, in_channels):
        super().__init__()
        # net_string = net_string.format(embedding_size=output_neurons)  # this is a bit hacky; should standardize to
        # output_neurons
        self.layers = layer_utils.add_layers_from_netstring(
            module=self, in_channels=in_channels, net_string=net_string, in_width=in_width, in_height=in_height)

    def forward(self, x):
        for layer in self.layers:
            # print('layer', layer)
            x = layer(x)
        return x


class DiscreteSender(nn.Module):
    """
    input: thought vector
    output: discrete utterance

    differentiable
    """

    def __init__(self, tau, vocab_size, embedding_size, utterance_max):
        super().__init__()
        self.decoder = language_model.GumbelDecoder(
            vocab_size=vocab_size, embedding_size=embedding_size, utterance_max=utterance_max, tau=tau,
            no_terminator=True)

    def forward(self, thought_vector):
        utterance, N = self.decoder(thought_vector)
        return utterance, N, 0


class DiscreteReceiver(nn.Module):
    """
    input: discrete utterance
    output: thought vector

    differentiable
    """
    def __init__(self, vocab_size, embedding_size):
        super().__init__()
        self.encoder = language_model.OnehotEncoder(
            vocab_size=vocab_size, embedding_size=embedding_size, no_terminator=True)

    def forward(self, utterance):
        thought_vector = self.encoder(utterance)
        return thought_vector


class DifferentiableActionValues(nn.Module):
    """
    input: state features
    output: list of value of actions over given action space, with one element per action, and
            one tensor per action dimension
    """
    def __init__(self, embedding_size, action_space):
        super().__init__()
        self.action_space = action_space
        self.embedding_size = embedding_size

        self.fcs = []
        for i, d in enumerate(action_space):
            fc = nn.Linear(embedding_size, d)
            self.fcs.append(fc)
            self.__setattr__('fc%s' % i, fc)
        self.action_ds = len(self.fcs)

    def forward(self, thought_vector):
        action_values = []
        for i, d in enumerate(self.action_space):
            fc = self.fcs[i]
            action_logits = fc(thought_vector)
            action_values.append(action_logits)
        return action_values


class StochasticActionsSelectorBasic(nn.Module):
    """
    input: input should already have same number of features as required output actions
    and should already have been softmaxed (ie its a probability distribution)
    output: actions over given action space
    """
    def __init__(self):
        super().__init__()
        # self.action_space = action_space
        # self.embedding_size = embedding_size

    def forward(self, x, global_idxes, testing=False):
        batch_size = x.size()[0]

        # elig_idxes_pairs = []  # mask is global_idxes
        # entropy = 0

        # actions = torch.LongTensor(batch_size).fill_(0)
        # action_probs_l = []
        matches_argmax_count = 0
        stochastic_draws_count = 0

        # action_logits = fc(thought_vector)
        action_probs = x
        # action_probs = F.softmax(action_logits)

        _, res_greedy = action_probs.data.max(1)
        res_greedy = res_greedy.view(-1, 1).long()

        log_g = None
        if testing:
            a = res_greedy
        else:
            m = Categorical(probs=action_probs)
            a = m.sample()
            log_g = m.log_prob(a)

        matches_argmax = res_greedy == a
        matches_argmax_count += matches_argmax.int().sum()
        stochastic_draws_count += batch_size

        action_probs_ = action_probs + 1e-8
        entropy = - (action_probs_ * action_probs_.log()).sum()

        res = {
            'action_probs': action_probs,
            'actions': a.cpu(),
            'entropy': entropy,
            'log_g': log_g,
            'global_idxes': global_idxes,
            'matches_argmax_count': matches_argmax_count,
            'stochastic_draws_count': stochastic_draws_count
        }
        return res
        # return action_probs, elig_idxes_pairs, actions, entropy, (matches_argmax_count, stochastic_draws_count)


class StochasticActionsSelector(nn.Module):
    """
    input: thoughtvector
    output: actions over given action space
    """
    def __init__(self, embedding_size, action_space):
        super().__init__()
        self.action_space = action_space
        self.embedding_size = embedding_size

        self.fcs = []
        for i, d in enumerate(action_space):
            fc = nn.Linear(embedding_size, d)
            self.fcs.append(fc)
            self.__setattr__('fc%s' % i, fc)
        self.enable_cuda = False

    def cuda(self):
        print('enabling cuda for StochasticActionsSelector')
        self.enable_cuda = True
        return self

    def forward(self, thought_vector, global_idxes, testing=False):
        batch_size = thought_vector.size()[0]

        elig_idxes_pairs = []  # mask is global_idxes
        entropy = 0

        actions = torch.LongTensor(batch_size, len(self.fcs)).fill_(0)
        action_probs_l = []
        matches_argmax_count = 0
        stochastic_draws_count = 0
        for i, fc in enumerate(self.fcs):
            action_logits = fc(thought_vector)
            action_probs = F.softmax(action_logits, dim=-1)

            _, res_greedy = action_probs.data.max(1)
            res_greedy = res_greedy.view(-1, 1).long()

            log_g = None
            if testing:
                a = res_greedy
            else:
                m = Categorical(probs=action_probs)
                a = m.sample()
                log_g = m.log_prob(a)

            matches_argmax = res_greedy == a
            matches_argmax_count += matches_argmax.int().sum()
            stochastic_draws_count += batch_size

            action_probs_ = action_probs + 1e-8
            entropy -= (action_probs_ * action_probs_.log()).sum()
            action_probs_l.append(action_probs.data)
            actions[:, i] = a.cpu()
            elig_idxes_pairs.append({
                'elig': log_g,
                'global_idxes': global_idxes
            })
        if self.enable_cuda:
            actions = actions.cuda()
        return action_probs_l, elig_idxes_pairs, actions, entropy, (matches_argmax_count, stochastic_draws_count)
