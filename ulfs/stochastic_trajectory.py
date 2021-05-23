"""
Used to store trajectory from policy gradients; can calc loss on this

concretely it comprises conceptually the log probabilities of the chosen actions

The log probabilities are autograd Variables. If the final reward is positive, we want
to increase the probability, and if the reward is negative we will want to reduce the probability.

The loss is calculated as the negative of the final reward multiplied by the log probability of the chosen
action. It is summed over all actions per example, and over all examples in the minibatch, then
backpropagated.

Since log_g is already indexed by the chosen action, we don't need to carry the chosen action itself
around, just log_g itself.


We can combine stochastic trajectories together by simply adding them. It will combine the various
log action probability tensors into a longer list of them. We'll ultimately calculate the loss
over all these tensors, by multiplying by the finnal reward, and taking the negative.

we need the global_idxes, because this will be used to index into the global rewards later
the global rewards will be over the full batch_size, and for each of the log_g tensors in the trajectory,
there is an associated global_idxes long tensor, which says which subset of the global rewards these log
probabilities are associated with

(in a sense batch_idxes might be a better name...)


that said, whilst we're at it, let's carry around the actions, the actionn probabilites, and the entropy too
"""


class StochasticTrajectory(object):
    def __init__(self, log_g=None, global_idxes=None, actions=None, entropy=None, action_probs=None):
        self.log_g_l = []
        self.global_idxes_l = []
        self.actions_l = []
        self.action_probs_l = []
        self.entropy_sum = 0  # no need to keep this split out
        if log_g is not None:
            assert actions is not None
            assert entropy is not None   # 0 ok ...
            assert action_probs is not None
            self.log_g_l.append(log_g)
            self.global_idxes_l.append(global_idxes)
            self.actions_l.append(actions)
            self.action_probs_l.append(action_probs)
            self.entropy_sum += entropy

    @property
    def entropy(self):
        return self.entropy_sum

    def append(self, log_g, global_idxes, actions, entropy):
        self.log_g_l.append(log_g)
        self.global_idxes_l.append(global_idxes)
        self.actions_l.append(actions)
        self.entropy_sum += entropy

    @classmethod
    def from_stochastic_sample(self, s):
        st = StochasticTrajectory()
        st.append_stochastic_sample(s=s)
        return st

    def append_stochastic_sample(self, s):
        self.log_g_l.append(s.log_g)
        self.global_idxes_l.append(s.batch_idxes)
        self.actions_l.append(s.actions)
        self.action_probs_l.append(s.action_probs)
        self.entropy_sum += s.entropy

    def __iadd__(self, second):
        self.log_g_l += second.log_g_l
        self.global_idxes_l += second.global_idxes_l
        self.actions_l += second.actions_l
        self.entropy_sum += second.entropy_sum
        return self

    def __add__(self, second):
        res = StochasticTrajectory()
        res += self
        res += second
        return res

    def calc_loss(self, rewards):
        loss = 0
        for i, log_g in enumerate(self.log_g_l):
            _global_idxes = self.global_idxes_l[i]
            _rewards = rewards
            if _global_idxes is not None:
                _rewards = rewards[_global_idxes]
            loss -= (_rewards * log_g).sum()
        return loss

    def debug_dump_contents(self):
        print('num things', len(self.log_g_l), len(self.global_idxes_l))
        for i in range(len(self.log_g_l)):
            print('i', i)
            print('  size log_g[i]', self.log_g_l[i].size())
            print('  size global_idxes_l[i]', self.global_idxes_l[i].size())
