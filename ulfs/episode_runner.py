"""
wrapper for eg worlds_container, or other env, that stores trajectory for q-learning
"""


from ulfs.trajectory import Trajectory


class EpisodeRunner(object):
    """
    stores trajectory. converts everything to cpu() before storing
    """
    def __init__(self, env):
        self.env = env
        # self.reset()

    def reset(self, seeds):
        self.states = self.env.reset(seeds=seeds)

        self.states_l = []
        self.actions_l = []
        self.rewards_l = []
        self.next_states_l = []
        self.dead_masks_l = []
        self.alive_masks_l = []
        self.done = False
        self._trajectory = None

        return self.states

    def act(self, actions):
        if self.states.is_cuda:
            self.states_l.append(self.states.cpu())
            self.actions_l.append(actions.cpu())
        else:
            self.states_l.append(self.states.clone())
            self.actions_l.append(actions.clone())

        rewards, dead_mask, next_states, done = self.env.act(actions)

        if next_states.is_cuda:
            self.rewards_l.append(rewards.cpu())
            self.dead_masks_l.append(dead_mask.cpu())
            self.alive_masks_l.append(1 - dead_mask.cpu())
            self.next_states_l.append(next_states.cpu())
        else:
            self.rewards_l.append(rewards.clone())
            self.dead_masks_l.append(dead_mask.clone())
            self.alive_masks_l.append(1 - dead_mask.clone())
            self.next_states_l.append(next_states.clone())
        self.done = done
        return rewards, dead_mask, next_states, done

    def next_timestep(self):
        self.states = self.env.next_timestep()
        return self.states

    @property
    def trajectory(self):
        # if self._trajectory is None:
        if True:
            self._trajectory = Trajectory(
                states_l=self.states_l,
                actions_l=self.actions_l,
                rewards_l=self.rewards_l,
                next_states_l=self.next_states_l,
                dead_masks_l=self.dead_masks_l,
                alive_masks_l=self.alive_masks_l
            )
        return self._trajectory
