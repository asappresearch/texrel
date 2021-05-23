import torch
from torch.distributions.categorical import Categorical


g_enable_cuda = False


def cudarize(something):
    """
    so we can simulate the cloning aspect of cudarizing
    """
    global g_enable_cuda
    if something is None:
        return None
    if g_enable_cuda:
        return something.cuda()
    else:
        return something.clone()


def unflatten_action(flat, action_space):
    """
    input: action numbers, in a batch, like:
        4
        3
        11
        ...
    output: split into axes:
        [1][1]
        [0][3]
        ...
    """
    batch_size = flat.size()[0]
    D = len(action_space)
    unflat = torch.LongTensor(batch_size, D)
    for d in range(D):
        dim = action_space[d]
        unflat[:, d] = flat % dim
        flat = flat / dim
    return unflat


def action_space_to_num_actions(action_space):
    res = 1
    for dim in action_space:
        res *= dim
    return res


def view_by_tensor(target, pos):
    view = target
    for d, i in enumerate(pos.tolist()):
        view = view.narrow(d, i, 1)
    return view


def calc_topk_intersect(a, b):
    """
    assumes that a and b are each a matrix, where rows are examples, and columns
    are various features

    Takes b as ground truth, determines for each example, how many features are
    tied for first/maximum place. This is ks

    Take the top k features of each example of a, and determines the proportion
    of these features which match the top k b features of the same example

    returns a score per example ([0, 1])
    """
    # device = 'cuda' if a.is_cuda else 'cpu'
    tensor_base = torch.cuda if a.is_cuda else torch
    # tensor_constr = torch.cuda.FloatTensor if a.is_cuda else torch.FloatTensor

    b_max, _ = b.max(dim=1)
    top_b_mask = ((b - b_max.view(-1, 1)).abs() < 1e-8)
    ks = top_b_mask.sum(dim=1)
    zc = tensor_base.FloatTensor(*a.size()).fill_(-1).cumsum(dim=1)
    k_mask = zc + ks.view(-1, 1).float() > -1

    _, a_sorted_idxes = a.sort(dim=1, descending=True)
    top_a_mask = tensor_base.ByteTensor(*a.size()).zero_().scatter_(1, a_sorted_idxes, k_mask)

    matches = (top_a_mask * top_b_mask).int().sum(dim=1)
    score = matches.float() / ks.float()
    return score


class CategoricalSample(object):
    def __init__(self, greedy_matches, actions, log_g, action_probs, entropy, batch_idxes):
        self.greedy_matches = greedy_matches
        self.actions = actions
        self.log_g = log_g
        self.action_probs = action_probs
        self.entropy = entropy
        self.batch_idxes = batch_idxes

    def calc_loss(self, rewards):
        if self.batch_idxes is not None:
            rewards = rewards[self.batch_idxes]
        loss = - (rewards * self.log_g).sum()
        return loss

    def __str__(self):
        return 'CategoricalSample(actions.size()=%s, actions.max=%s)' % (self.actions.size(), self.actions.max().item())


def draw_categorical_sample(action_probs, batch_idxes, training=True):
    """
    uses the last dimension of action_probs as the probailbity distribution
    """
    _, a_greedy = action_probs.max(dim=-1)
    if training:
        m = Categorical(probs=action_probs)
        a = m.sample()
        log_g = m.log_prob(a)

        action_probs_ = action_probs + 1e-8
        entropy = - (action_probs_ * action_probs_.log()).sum()
        greedy_matches = (a_greedy == a).float().mean().item()

        return CategoricalSample(
            greedy_matches=greedy_matches, actions=a, log_g=log_g, action_probs=action_probs, entropy=entropy,
            batch_idxes=batch_idxes)
    else:
        a = a_greedy

        # action_probs_ = action_probs + 1e-8
        # entropy = - (action_probs_ * action_probs_.log()).sum()
        return CategoricalSample(
            greedy_matches=1.0, actions=a, log_g=None, action_probs=action_probs, entropy=0, batch_idxes=batch_idxes)
