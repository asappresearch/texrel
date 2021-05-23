"""
Will record action probabilities and utterances for a number of steps,
and use this to determine lexicon size per action over time
"""
import torch
import numpy as np
from collections import defaultdict
import scipy
import scipy.stats


def utterances_tensors_to_utterance_ints(utterances, utterances_lens):
    """
    converts the utterances in the incoming tensors to a list of integers, eg 15235, 111222, etc
    Note that since 0 terminates an utterance, we dont need to worry about leading zeros.
    the empty utterance will be simply 0

    assumes incoming utterances tensor is zero padded, ie no undefined values

    zero padding is assumed to have been
    added on the right hand side of each row, ie rows will look something like:
        [ 3,  2,  1,  3,  1,  0],
        [ 1,  0,  0,  0,  0,  0],
        [ 1,  2,  2,  2,  1,  0],
        [ 1,  3,  0,  0,  0,  0],
        [ 2,  3,  0,  0,  0,  0],
    """
    N = utterances.size()[0]
    max_len = utterances.size()[1]
    assert len(utterances.size()) == 2
    assert len(utterances_lens.size()) == 1
    utterances_ints_by_len = torch.LongTensor(N, max_len)
    if utterances.is_cuda:
        utterances_ints_by_len = utterances_ints_by_len.cuda()
    utterances_ints_by_len[:, 0] = utterances[:, 0]
    for t in range(1, max_len):
        utterances_ints_by_len[:, t] = utterances_ints_by_len[:, t - 1] * 10 + utterances[:, t]
    utterances_ints = []
    for n in range(N):
        _len = min(utterances_lens[n], max_len - 1)
        utterances_ints.append(utterances_ints_by_len[n, _len].item())
    return utterances_ints


class LexiconRecorder(object):
    def __init__(self, num_actions):
        self.num_actions = num_actions
        self.reset()

    def record(self, action_probs_l, utterances_by_t, utterance_lens_by_t):
        """
        incoming actions and utterances are for a single episode
        - there will be as many tensors in each list as timesteps in the episode
        - each member of each list is a tensor
        - the size of the tensors will decrease as games end, so we dont need to sieve these tensors:
          they are already sieved
        """
        T = len(action_probs_l)
        assert len(utterances_by_t) == T
        assert len(utterance_lens_by_t) == T
        for t in range(T):
            action_probs = action_probs_l[t]
            utterances = utterances_by_t[t]
            utterances_lens = utterance_lens_by_t[t]
            _, greedy_actions = action_probs.max(-1)
            utterances_ints = utterances_tensors_to_utterance_ints(utterances, utterances_lens)
            N = utterances.size()[0]
            for n in range(N):
                self.count_by_utterance_by_action[greedy_actions[n]][utterances_ints[n]] += 1

    def print_lexicon(self):
        """
        only print most frequent usages
        we want most frequent per action; and number of others
        """
        for action, count_by_utterance in enumerate(self.count_by_utterance_by_action):
            utt_counts = [{'utt': utt, 'count': count} for utt, count in count_by_utterance.items()]
            utt_counts.sort(key=lambda x: x['count'], reverse=True)
            # this is pretty hacky. 0 is the null terminator...
            # and this cant handle vocab size more than 8 ish
            utts_str = ' '.join([str(x['utt']).replace('0', '') + '=' + str(x['count']) for x in utt_counts[:5]])
            print('l[:5]', utts_str)

    def reset(self):
        self.count_by_utterance_by_action = []
        for n in range(self.num_actions):
            self.count_by_utterance_by_action.append(defaultdict(int))

    def calc_stats(self):
        total_utt = 0
        total_unique = 0
        num_utterances_by_a = []
        unique_by_a = []
        for a in range(self.num_actions):
            a_utterances = np.sum(list(self.count_by_utterance_by_action[a].values())).item()
            a_unique = len(self.count_by_utterance_by_action[a])
            total_utt += a_utterances
            total_unique += a_unique
            num_utterances_by_a.append(a_utterances)
            unique_by_a.append(a_unique)
        stats = {
            'median_unique_utt': np.median(unique_by_a).item(),
            'mean_unique_utt': np.mean(unique_by_a).item(),
            'min_unique_utt': np.min(unique_by_a).item(),
            'max_unique_utt': np.max(unique_by_a).item(),
            'unique_utt_ent': scipy.stats.entropy(unique_by_a).item(),
            'total_utt_ent': scipy.stats.entropy(num_utterances_by_a).item(),
            'total_utt': total_utt,
            'total_unique': total_unique
        }
        return stats
