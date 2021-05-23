from ulfs import alive_sieve


class Trajectory(object):
    """
    You can assign additional lists to this, which will end up in the iterated experiences
    the name of the item in the experience will be the name the list is assigned to with
    's_l' removed, eg 'states_l' => 'state'
    """
    def __init__(self, states_l, actions_l, rewards_l, dead_masks_l, next_states_l, alive_masks_l):
        self.keys = []
        self.states_l = states_l
        self.actions_l = actions_l
        self.rewards_l = rewards_l
        self.next_states_l = next_states_l
        self.dead_masks_l = dead_masks_l
        self.alive_masks_l = alive_masks_l

    def __setattr__(self, k, v):
        self.__dict__[k] = v
        if k not in ['keys', 'next_states_l']:
            self.keys.append(k)

    def iter_experiences(self):
        if len(self.states_l) == 0:
            return

        # enable_cuda = self.states_l[0].is_cuda
        enable_cuda = False
        sieve_playback = alive_sieve.SievePlayback(self.alive_masks_l, enable_cuda=enable_cuda)
        for t, global_idxes in sieve_playback:
            _next_states = self.next_states_l[t]
            _alive_mask = sieve_playback.alive_mask
            for i in range(sieve_playback.batch_size):
                experience = {'next_state': None}
                for k in self.keys:
                    v = getattr(self, k)
                    experience[k.replace('s_l', '')] = v[t][i]
                if _alive_mask[i]:
                    experience['next_state'] = _next_states[i]
                yield experience

    def iter_batches(self):
        if len(self.states_l) == 0:
            return

        # enable_cuda = self.states_l[0].is_cuda
        enable_cuda = False
        sieve_playback = alive_sieve.SievePlayback(self.alive_masks_l, enable_cuda=enable_cuda)
        for t, global_idxes in sieve_playback:
            _next_states = self.next_states_l[t]
            _alive_mask = sieve_playback.alive_mask
            batch = {}
            batch['next_states'] = _next_states
            batch['alive_mask'] = _alive_mask
            batch['global_idxes'] = global_idxes
            for k in self.keys:
                v = getattr(self, k)
                batch[k.replace('_l', '')] = v[t]
            yield batch
