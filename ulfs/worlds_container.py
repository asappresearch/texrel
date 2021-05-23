"""
Groups a batch_size block of worlds together; can run act, reset etc on entire batch
"""
import torch

from ulfs import alive_sieve
from ulfs.rl_common import cudarize


class WorldsContainer(object):
    """
    Contains a bunch of worlds, runs action tensor against them, and returns
    rewards, dead_mask, and next_states tensor

    public properties:
    - global_rewards C
    - states C
    - done
    - timesteps C
    - rewards C
    - alive_masks C
    ('C' means 'cudarized by .cuda()')
    """
    def __init__(self, world_constructor, world_constructor_kwargs, batch_size, max_steps):
        self.all_worlds = []
        self.batch_size = batch_size
        self.enable_cuda = False
        self.max_steps = max_steps
        for n in range(batch_size):
            # print('worlds constructor seed', seed)
            # if seed is not None:
            #     world_constructor_kwargs['seed'] = seed + n
            world = world_constructor(**world_constructor_kwargs)
            self.all_worlds.append(world)
        self.type_constr = torch
        self.reset(seeds=list(range(self.batch_size)))

    def cuda(self):
        self.enable_cuda = True
        self.type_constr = torch.cuda
        return self

    def reset(self, seeds):
        # print('worlds_container.reset()')
        self.worlds = list(self.all_worlds)
        self.sieve = alive_sieve.AliveSieve(
            batch_size=self.batch_size, enable_cuda=self.enable_cuda)
        self.global_timesteps = torch.LongTensor(self.batch_size).fill_(self.max_steps)
        self.global_rewards = self.type_constr.FloatTensor(self.batch_size).fill_(0)  # full-length, never sieved
        # self.timesteps = cudarize(self.timesteps)
        if self.enable_cuda:
            self.global_timesteps = self.global_timesteps.cuda()
        states = torch.zeros(
            self.batch_size, *self.worlds[0].state_size)
        for b in range(self.batch_size):
            states[b] = self.worlds[b].reset(seed=None if seeds is None else seeds[b])
        # states = cudarize(states)
        # self.global_rewards = cudarize(self.global_rewards)
        if self.enable_cuda:
            states = states.cuda()
            self.global_rewards = self.global_rewards.cuda()
        self.t = 0
        self.states = states
        self.done = False
        self.alive_masks = []
        return states

    def act(self, actions):
        """
        actions should be 1-dimensional, can be cuda (we'll cpu it)
        """
        actions = actions.cpu()
        batch_size = self.sieve.batch_size
        rewards = torch.FloatTensor(batch_size).fill_(0)
        dead_mask = torch.ByteTensor(batch_size).fill_(0)
        states = torch.zeros(batch_size, *self.worlds[0].state_size)
        for b in range(batch_size):
            _render = False
            # if render and sieve.alive_idxes[0] == 0:
            #     _render = True
            # loc = worlds[b].agent_loc
            # positions_visited[b][loc[0], loc[1]] = 1
            _reward, _done = self.worlds[b].act(actions[b].item(), render=_render)
            rewards[b] = _reward
            dead_mask[b] = int(_done)
            states[b] = self.worlds[b].get_state()
        self.sieve.mark_dead(dead_mask)
        self.alive_masks.append(self.sieve.alive_mask.clone())
        dead_idxes = self.sieve.get_dead_idxes()
        if len(dead_idxes) > 0:
            self.global_timesteps[self.sieve.global_idxes[dead_idxes]] = self.t + 1
        rewards = cudarize(rewards)
        dead_mask = cudarize(dead_mask)
        states = cudarize(states)

        if self.enable_cuda:
            rewards = rewards.cuda()
            states = states.cuda()
            dead_mask = dead_mask.cuda()

        self.global_rewards[self.sieve.global_idxes] += rewards
        self.rewards = rewards
        self.dead_mask = dead_mask
        self.states = states
        self.done = self.sieve.all_dead()
        return rewards, dead_mask, states, self.done

    def next_timestep(self):
        self.worlds = self.sieve.sieve_list(self.worlds)
        self.states = self.sieve.sieve_tensor(self.states)
        self.sieve.self_sieve_()
        self.rewards = None
        self.dead_mask = None
        self.t += 1
        return self.states

    @property
    def global_idxes(self):
        return self.sieve.global_idxes

    @property
    def state_size(self):
        return self.worlds[0].state_size

    @property
    def action_space(self):
        return self.worlds[0].action_space

    def get_int_tensor(self, attribute_name):
        values = [getattr(world, attribute_name) for world in self.worlds]
        res = torch.IntTensor(values)
        return res

    def get_full_int_tensor(self, attribute_name):
        values = [getattr(world, attribute_name) for world in self.all_worlds]
        res = torch.IntTensor(values)
        return res
