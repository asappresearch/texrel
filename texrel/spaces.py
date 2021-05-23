# import random
from typing import Optional, List, Tuple
import numpy as np

from texrel import things, relations


class SpaceNoValuesAvailable(Exception):
    pass


class IntSpace(object):
    def __init__(self, r: np.random.RandomState, num_ints: int, avail_ints: Optional[List[int]] = None) -> None:
        """
        Parameters
        ----------
        num_ints: int
            used to decide onehot_size, and to populate avail_ints if not provided
        avail_ints: Optional[List[int]]
            the ints in this space. if not provided, then all possible ints from
            0 to num_ints will be included
        """
        self.r = r
        self.num_ints = num_ints
        self.size = self.num_ints
        self.onehot_size = self.num_ints
        self.sizes_l = [self.num_ints]
        if avail_ints is None:
            self.avail_ints = list(range(num_ints))
        else:
            self.avail_ints = avail_ints

    def __str__(self) -> str:
        return f'IntSpace(num_ints={self.num_ints},avail_ints=[' + ','.join(
            [str(v) for v in self.avail_ints]) + '])'

    def __len__(self) -> int:
        return len(self.avail_ints)

    def sample(self) -> int:
        return self.r.choice(self.avail_ints).item()

    def partition(self, sizes: List[int]):
        """
        partition this space into sizes in sizes
        sum(sizes) should equal the length of this space
        """
        N = len(self)
        assert sum(sizes) == N
        idxes = self.r.choice(N, N, replace=False)
        shuffled = [self.avail_ints[idx] for idx in idxes]
        returned_spaces = []
        start_idx = 0
        for i, N_sub in enumerate(sizes):
            child_avail = shuffled[start_idx: start_idx + N_sub]
            child_space = IntSpace(r=self.r, num_ints=self.num_ints, avail_ints=child_avail)
            returned_spaces.append(child_space)
            start_idx += N_sub
        return returned_spaces


class MultiIntUnorderedSpace(object):
    def __init__(
            self,
            r: np.random.RandomState,
            # dims: List[int],
            num_dims: int,
            num_ints: int,
            avail_ints: Optional[List[Tuple[int, ...]]] = None) -> None:
        """
        Parameters
        ----------
        num_ints: int
            used to decide onehot_size, and to populate avail_ints if not provided
        avail_ints: Optional[List[int]]
            the ints in this space. if not provided, then all possible ints from
            0 to num_ints will be included
        """
        self.r = r
        self.num_dims = num_dims
        self.num_ints = num_ints
        self.size = self.num_ints
        self.onehot_size = self.num_ints
        self.sizes_l = [self.num_ints]
        if avail_ints is None:
            self.avail_ints = []
            for ints in np.ndindex(*([num_ints] * num_dims)):
                if sorted(ints) == list(ints):
                    self.avail_ints.append(ints)
        else:
            self.avail_ints = avail_ints
        self.avail_ints_set = set(self.avail_ints)

    def __str__(self) -> str:
        if len(self) < 30:
            return f'MultiIntUnorderedSpace(num_dims={self.num_dims},num_ints={self.num_ints},avail_ints=[' + ','.join(
                [str(v) for v in self.avail_ints]) + '])'
        else:
            return f'MultiIntUnorderedSpace(num_ints={self.num_ints},avail_ints=[' + ','.join(
                [str(v) for v in self.avail_ints[:30]]) + ',...])'

    def __contains__(self, ints: Tuple[int]) -> bool:
        return ints in self.avail_ints_set

    def __len__(self) -> int:
        return len(self.avail_ints)

    def sample(self) -> Tuple[int]:
        idx = self.r.randint(len(self.avail_ints))
        return self.avail_ints[idx]

    def resample_dim(self, ints: List[int], dim: int) -> Tuple[int, ...]:
        """
        given a valid sample, and a dimension index, resamples a value for that dimension,
        such that the new sample is contained in the space
        assumes that the space is large enough that there is another valid value available for
        that dim
        """
        new_ints = list(ints)
        old_v = ints[dim]
        new_v = old_v
        tries = 0

        while new_v == old_v or tuple(sorted(new_ints)) not in self.avail_ints_set:
            new_v = self.r.randint(self.num_ints)
            new_ints[dim] = new_v
            tries += 1
            if tries > 5000:
                print('num_ints', self.num_ints, 'num_dims', self.num_dims, 'old_v', old_v, 'dim', dim)
                raise SpaceNoValuesAvailable()
        return tuple(sorted(new_ints))

    def partition(self, sizes: List[int]):
        """
        partition this space into sizes in sizes
        sum(sizes) should equal the length of this space
        """
        N = len(self)
        assert sum(sizes) == N
        idxes = self.r.choice(N, N, replace=False)
        shuffled = [self.avail_ints[idx] for idx in idxes]
        returned_spaces = []
        start_idx = 0
        for i, N_sub in enumerate(sizes):
            child_avail = shuffled[start_idx: start_idx + N_sub]
            child_space = MultiIntUnorderedSpace(
                r=self.r, num_dims=self.num_dims, num_ints=self.num_ints, avail_ints=child_avail)
            returned_spaces.append(child_space)
            start_idx += N_sub
        return returned_spaces


class ThingSpace(object):
    def __init__(
            self, r: np.random.RandomState, color_space: IntSpace, shape_space: IntSpace,
            available_items: Optional[List[things.ShapeColor]] = None) -> None:
        self.r = r
        self.color_space = color_space
        self.shape_space = shape_space
        self.onehot_size = self.color_space.onehot_size + self.shape_space.onehot_size
        self.sizes_l = self.shape_space.sizes_l + self.color_space.sizes_l
        if available_items is None:
            available_items = []
            for color in range(color_space.size):
                for shape in range(shape_space.size):
                    shape_color = things.ShapeColor(shape=shape, color=color)
                    available_items.append(shape_color)
        self.available_items = available_items
        self.available_items_set = set(self.available_items)

    def __len__(self) -> int:
        return len(self.available_items)

    def __contains__(self, o: things.ShapeColor) -> bool:
        return o in self.available_items_set

    def sample(self) -> things.ShapeColor:
        item_idx = self.r.randint(len(self.available_items))
        return self.available_items[item_idx]

    def __str__(self) -> str:
        if len(self.available_items) < 10:
            return f'ThingSpace(available_items={self.available_items})'
        res = 'ThingSpace(' + ','.join([str(o) for o in self.available_items]) + ')'
        return res

    @property
    def num_unique_things(self) -> int:
        return len(self.available_items)

    def partition(self, partition_sizes: List[int]) -> List['ThingSpace']:
        """
        returns new ThingSpaces. Each ThingSpace is for a partition_size'd
        subset of this thingspace. All thingspaces are disjoint
        """
        assert np.sum(partition_sizes).item() == len(self.available_items)

        # shuffled_items = random.sample(self.available_items, len(self.available_items))
        shuffled_items = list(self.r.choice(self.available_items, len(self.available_items), replace=False))

        new_spaces = []
        pos = 0
        for i, partition_size in enumerate(partition_sizes):
            items = shuffled_items[pos: pos + partition_size]
            pos += partition_size
            new_space = ThingSpace(
                r=self.r,
                color_space=self.color_space,
                shape_space=self.shape_space,
                available_items=items
            )
            new_spaces.append(new_space)
        return new_spaces


class PrepositionSpace(object):
    def __init__(self, r: np.random.RandomState, available_preps=None):
        self.r = r
        self.prepositions = relations.prepositions
        if available_preps is not None:
            self.prepositions = []
            for prep in available_preps:
                self.prepositions.append(getattr(relations, prep))
        self.id_by_preposition = {preposition: i for i, preposition in enumerate(self.prepositions)}
        self.num_prepositions = len(self.prepositions)
        self.size = self.num_prepositions
        self.onehot_size = self.num_prepositions
        self.sizes_l = [self.num_prepositions]

    def sample(self):
        id = self.r.randint(0, self.num_prepositions)
        return self.prepositions[id]()


class RelationSpace(object):
    def __init__(self, thing_space: ThingSpace, prep_space: PrepositionSpace):
        self.thing_space = thing_space
        self.prep_space = prep_space
        self.onehot_size = 2 * self.thing_space.onehot_size + self.prep_space.onehot_size
        self.sizes_l = self.thing_space.sizes_l + self.prep_space.sizes_l + self.thing_space.sizes_l

    def sample(self):
        thing1 = self.thing_space.sample()
        thing2 = None
        tries = 0
        while thing2 is None or thing2 == thing1:
            thing2 = self.thing_space.sample()
            tries += 1
            if tries > 5000:
                print('thing1', thing1)
                print('self.thing_space.available_items', self.thing_space.available_items)
                raise SpaceNoValuesAvailable()
        prep = self.prep_space.sample()
        return relations.Relation(left=thing1, prep=prep, right=thing2)
