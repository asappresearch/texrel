"""
We want random draws to be deterministic, based on things like:
- some underlying seed, that we provide to the program, like 351413, or 123, or whatever
  - this will henceforth be called the 'global' seed
- episode number (so we dont have to iterate across all stochastic draws through all episodes, when we load)
  - (and perhaps the batch number, but I think I'm currently saving only at the end of a full episode)
- the thing we are drawing (exploration, world reset, dropout, ...), so we dont get correlation across
  these things
- weight initialization
  - will only happen for the first episode, so just needs to be a function of the global seed
- to faciliate this, making a class to handle it :)

Assumptions:
- we'll only want to run against up to 1000 possible different global seeds
- there are only 10 'things' we want to draw random numbers for, eg indexes, exploration, etc

Both random and torch
- we'll sometimes want to draw numbers using torch, not just using numpy random generator
- including for CUDA
- => implies we might just want to seed these, in a deterministic way, based on some function
  of global seed etc, and not generate these random numbers directly from numpy.random

concept:
- we'll create a numpy random number generator, based on the global seed and the episode number
- this will be used to provide random seeds for eg initializing weights etc

The seed generator random instance will be a python mersenne twister instance, based on the seed and the episode number,
something like:
    np.random.RandomState((episode_number * 1000 + global_seed) * 10 + thing-index)

we're going to include the various world indexes in the 'things' by doing like eg 'world1', 'world2', ...

revision: we just generate the seeds directly, no random state involved


GOTCHAS
- random number generation gives different numbers on gpu than on cpu EVEN WITH THE SAME SEED
- therefore, for consistency between cpu and gpu, either always run on gpu, or,
- generate random numbers always on cpu, then copy to gpu

Given that dropout etc make generating numbers always on cpu hard, so personally I'm going to go wtih:
- only guarantee reproducibility conditional on choice of gpu vs cpu
"""
import numpy as np
import torch


class SeedManager(object):
    def __init__(self, global_seed=None, max_global_seeds=1000, max_things=256):
        self.max_global_seeds = max_global_seeds
        self.max_things = max_things

        if global_seed is None:
            global_seed = np.random.randint(0, max_global_seeds - 1)
        self.global_seed = global_seed
        assert global_seed < max_global_seeds
        print('using global seed', global_seed)
        self.things = []
        self.thing_idx_by_name = {}

    def _get_thing_idx(self, thing_name):
        idx = self.thing_idx_by_name.get(thing_name, None)
        if idx is not None:
            return idx

        assert len(self.thing_idx_by_name) < self.max_things
        idx = len(self.thing_idx_by_name) + 1
        self.things.append(thing_name)
        self.thing_idx_by_name[thing_name] = idx
        return idx

    def calc_seed(self, episode_number, thing_name):
        """
        thing_name could be eg 'world1', 'world2', 'weightsinit', 'exploreprob'
        """
        thing_idx = self._get_thing_idx(thing_name)
        seed = ((episode_number * self.max_global_seeds) + self.global_seed) * self.max_things + thing_idx
        return seed

    def calc_world_seeds(self, episode_number, num_worlds):
        """
        will return a list of seeds, based on thing names world0, world1, ...
        """
        seeds = []
        for i in range(num_worlds):
            seed = self.calc_seed(episode_number=episode_number, thing_name='world%s' % i)
            seeds.append(seed)
        return seeds

    def torch_manual_seed(self, episode_number, thing_name):
        """
        thing_name could be eg 'world1', 'world2', 'weightsinit', 'exploreprob'
        """
        seed = self.calc_seed(episode_number=episode_number, thing_name=thing_name)
        torch.manual_seed(seed)
        return seed

    def generate_np_seeded_state(self, episode_number, thing_name):
        """
        thing_name could be eg 'world1', 'world2', 'weightsinit', 'exploreprob'
        """
        seed = self.calc_seed(episode_number=episode_number, thing_name=thing_name)
        r = np.random.RandomState(seed)
        return r
