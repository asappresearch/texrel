"""
Various latent hypotheses, that we can use to generate data from
"""
import numpy as np
import torch
import abc
from typing import Tuple, List, Union, Any, Iterable

from texrel import things as things_lib, relations, spaces, grid as grid_lib
from texrel.grid import Grid


class Hypothesis(abc.ABC):
    def create_example(self, label: int, grid_size: int, num_distractors: int) -> grid_lib.Grid:
        """
        label is 1 for positive example, or 0 for negative example
        """
        if label == 1:
            return self.create_positive_example(grid_size=grid_size, num_distractors=num_distractors)
        elif label == 0:
            return self.create_negative_example(grid_size=grid_size, num_distractors=num_distractors)
        else:
            raise Exception('unhandled label', label)

    @abc.abstractmethod
    def create_positive_example(self, grid_size: int, num_distractors: int) -> grid_lib.Grid:
        pass

    @abc.abstractmethod
    def create_negative_example(self, grid_size: int, num_distractors: int) -> grid_lib.Grid:
        pass

    @abc.abstractmethod
    def as_seq(self) -> Tuple[torch.Tensor, List[str]]:
        pass

    @abc.abstractmethod
    def as_english(self) -> str:
        pass

    @abc.abstractmethod
    def as_english_structure(self) -> Union[str, Tuple[Any, Any]]:
        pass


class MultiIntUnorderedHypothesis(object):
    """
    doesnt handle distractors (they have to be added by something that knows about
    more than just ints), but can handle creating positive/negative examples of the ints
    themselves. without grid placement, or distractors.

    # we assume here that distractor_space is either the entire underlying space, or close to it

    we sample distractors, and negative examples, from the entire space,
    without regard to training vs holdout partitions.

    (since an object of this class already has the positive example ints assigned to it,
    at construction time, so it therefore doesnt need to consider training vs holdout
    partitions at all)
    """
    def __init__(self, r: np.random.RandomState, ints: Tuple[int, ...], dim_length: int):
        self.r = r
        self.ints = ints
        self.dim_length = dim_length

    def create_positive_example(self, num_distractors: int) -> Tuple[int, ...]:
        distractors: List[int] = []
        tries = 0
        while len(distractors) < num_distractors:
            v_dist = self.r.randint(self.dim_length)
            if v_dist not in self.ints:
                distractors.append(v_dist)
            tries += 1
            if tries > 5000:
                print('ints', self.ints, 'dim_length', self.dim_length)
                raise spaces.SpaceNoValuesAvailable()
        return tuple(sorted(list(self.ints) + distractors))

    def __eq__(self, b: object) -> bool:
        if not isinstance(b, self.__class__):
            print('not class')
            return False
        print(self.dim_length, b.dim_length)
        print(self.ints == b.ints, self.dim_length == b.dim_length)
        return self.ints == b.ints and self.dim_length == b.dim_length

    def sample_int(self) -> int:
        return self.r.randint(self.dim_length)

    def create_negative_example(self, num_distractors: int) -> Tuple[int, ...]:
        """
        we modify a single int
        """
        num_dims = len(self.ints)
        flip_dim = self.r.randint(num_dims)
        old_v = self.ints[flip_dim]
        new_ints = list(self.ints)
        new_v = old_v
        tries = 0
        while new_v == old_v:
            new_v = self.sample_int()
            tries += 1
            if tries > 5000:
                raise spaces.SpaceNoValuesAvailable()
        new_ints[flip_dim] = new_v

        tries = 0
        distractors: List[int] = []
        while len(distractors) < num_distractors:
            v_dist = self.sample_int()
            if v_dist not in self.ints:
                distractors.append(v_dist)
                tries = 0
                continue
            tries += 1
            if tries > 5000:
                print('ints', self.ints, 'dim_length', self.dim_length)
                raise spaces.SpaceNoValuesAvailable()

        return tuple(sorted(new_ints + distractors))

    def __str__(self):
        return self.__class__.__name__ + '(' + str(self.ints) + ')'


class RelationHypothesis(Hypothesis):
    """
    represents a single relations, and
    generates examples of this relation
    """
    def __init__(
            self,
            r: np.random.RandomState,
            relation: relations.Relation,
            rel_space: spaces.RelationSpace,
            distractor_thing_space: spaces.ThingSpace) -> None:
        self.r = r
        self.relation = relation
        self.rel_space = rel_space
        self.thing_space = rel_space.thing_space
        self.distractor_thing_space = distractor_thing_space

    def __str__(self) -> str:
        return 'relation: ' + str(self.relation)

    def __eq__(self, b: object) -> bool:
        if not isinstance(b, RelationHypothesis):
            return False
        return self.relation == b.relation

    def as_english(self) -> str:
        return self.relation.as_english()

    def as_english_structure(self) -> Union[str, Tuple[Any, Any]]:
        return self.relation.as_english_structure()

    def as_seq(self) -> Tuple[torch.Tensor, List[str]]:
        idxes, types = self.relation.as_indices(rel_space=self.rel_space)
        seq = torch.LongTensor(idxes)
        return seq, types

    @classmethod
    def from_seq(
            cls,
            r: np.random.RandomState,
            seq: torch.Tensor,
            rel_space: spaces.RelationSpace,
            distractor_thing_space: spaces.ThingSpace) -> 'RelationHypothesis':
        return cls(
            r=r,
            relation=relations.Relation.eat_from_indices(indices=seq.tolist(), rel_space=rel_space)[0],
            rel_space=rel_space,
            distractor_thing_space=distractor_thing_space
        )

    def _create_example(
            self,
            exclude_objects: List[things_lib.ShapeColor],
            relation: relations.Relation,
            grid_size: int,
            num_distractors: int) -> Grid:
        # r = self.r
        grid = Grid(size=grid_size)

        grid_size = grid.size

        first_h_range = [0, grid_size]
        first_w_range = [0, grid_size]
        second_h_range = [0, grid_size]
        second_w_range = [0, grid_size]
        prep = relation.prep
        if isinstance(prep, relations.RightOf):
            # eg first rightof second
            # means some constraints on the x range of each
            # eg if second is in right-most column, impossible for first to fit somewhere
            first_w_range = [1, grid_size]
        elif isinstance(prep, relations.LeftOf):
            first_w_range = [0, grid_size - 1]
        elif isinstance(prep, relations.Above):
            # ie first above second
            # assume h is downwards
            first_h_range = [0, grid_size - 1]
        elif isinstance(prep, relations.Below):
            first_h_range = [1, grid_size]
        elif prep.__class__ in [
                # relations.NotAbove, relations.NotBelow,
                # relations.NotRightOf, relations.NotLeftOf,
                relations.HorizSame, relations.VertSame]:
            # no constraints in fact; first can go anywhere
            pass
        else:
            raise Exception('preposition not handled ' + str(prep))

        h1 = self.r.randint(*first_h_range)
        w1 = self.r.randint(*first_w_range)

        # now figure out second ranges
        if isinstance(prep, relations.RightOf):
            # eg first rightof second
            # means some constraints on the x range of each
            # eg if second is in right-most column, impossible for first to fit somewhere
            second_w_range = [0, w1]
        elif isinstance(prep, relations.LeftOf):
            second_w_range = [w1 + 1, grid_size]
        elif isinstance(prep, relations.Above):
            # ie first above second
            # assume h is downwards
            second_h_range = [h1 + 1, grid_size]
        elif isinstance(prep, relations.Below):
            second_h_range = [0, h1]
        elif isinstance(prep, relations.VertSame):
            second_w_range = [w1, w1 + 1]
        elif isinstance(prep, relations.HorizSame):
            second_h_range = [h1, h1 + 1]
        else:
            raise Exception('preposition not handled ' + str(prep))
        h2 = h1
        w2 = w1
        while h2 == h1 and w2 == w1:
            h2 = self.r.randint(*second_h_range)
            w2 = self.r.randint(*second_w_range)

        o1 = relation.left
        o2 = relation.right

        grid.add_object((h1, w1), o1)
        grid.add_object((h2, w2), o2)

        distractors: List[things_lib.ShapeColor] = []
        exclude_objects = list(set(exclude_objects + grid.objects_set))
        for i in range(num_distractors):
            o = None
            while o is None or o in exclude_objects:
                o = self.distractor_thing_space.sample()
            distractors.append(o)

        for i, o in enumerate(distractors):
            pos = grid.generate_available_pos(r=self.r)
            grid.add_object(pos, o)

        return grid

    def create_positive_example(self, grid_size: int, num_distractors: int) -> Grid:
        return self._create_example(
            exclude_objects=[], relation=self.relation, grid_size=grid_size, num_distractors=num_distractors)

    def create_negative_example(self, grid_size: int, num_distractors: int) -> Grid:
        exclude_objects = [self.relation.left, self.relation.right]
        neg_types = ['preposition', 'left_color', 'right_color', 'left_shape', 'right_shape']

        def swap_color(old_o: things_lib.ShapeColor) -> things_lib.ShapeColor:
            new_o = old_o
            tries = 0
            while new_o == old_o or new_o not in self.distractor_thing_space:
                new_color = self.thing_space.color_space.sample()
                new_o = things_lib.ShapeColor(color=new_color, shape=old_o.shape)
                tries += 1
                if tries > 50:
                    raise spaces.SpaceNoValuesAvailable()
            return new_o

        def swap_shape(old_o: things_lib.ShapeColor) -> things_lib.ShapeColor:
            new_o = old_o
            tries = 0
            while new_o == old_o or new_o not in self.distractor_thing_space:
                new_shape = self.thing_space.shape_space.sample()
                new_o = things_lib.ShapeColor(color=old_o.color, shape=new_shape)
                tries += 1
                if tries > 50:
                    # print('old_o', old_o)
                    raise spaces.SpaceNoValuesAvailable()
            return new_o

        tries = 0
        while True:
            neg_type = self.r.choice(neg_types)
            try:
                if neg_type == 'preposition':
                    relation = self.relation.complement()
                elif neg_type == 'left_color':
                    new_o = swap_color(old_o=self.relation.left)
                    relation = relations.Relation(
                        left=new_o, prep=self.relation.prep, right=self.relation.right)
                elif neg_type == 'right_color':
                    new_o = swap_color(old_o=self.relation.right)
                    relation = relations.Relation(
                        left=self.relation.left, prep=self.relation.prep, right=new_o)
                elif neg_type == 'left_shape':
                    new_o = swap_shape(old_o=self.relation.left)
                    relation = relations.Relation(
                        left=new_o, prep=self.relation.prep, right=self.relation.right)
                elif neg_type == 'right_shape':
                    new_o = swap_shape(old_o=self.relation.right)
                    relation = relations.Relation(
                        left=self.relation.left, prep=self.relation.prep, right=new_o)
                break
            except spaces.SpaceNoValuesAvailable:
                pass
            if tries >= 100:
                print('relation', relation)
                print('self.distractor_thing_space.available_items', self.distractor_thing_space.available_items)
                print('self.thing_space.available_items', self.thing_space.available_items)
                raise spaces.SpaceNoValuesAvailable()
        return self._create_example(
            exclude_objects=exclude_objects, relation=relation, grid_size=grid_size, num_distractors=num_distractors)


class ThingsHypothesis(Hypothesis):
    def __init__(
            self,
            r: np.random.RandomState,
            things: List[things_lib.ShapeColor],
            thing_space: spaces.ThingSpace,
            distractor_thing_space: spaces.ThingSpace) -> None:
        self.r = r
        self.things = things
        self.thing_space = thing_space
        self.distractor_thing_space = distractor_thing_space

    def as_english(self) -> str:
        return 'has-shapecolors ' + ' '.join([f'color{o.color} shape{o.shape}' for o in self.things])

    def as_english_structure(self) -> Union[str, Tuple[Any, Any]]:
        if len(self.things) == 1:
            return ('has-shapecolors', (f'color{self.things[0].color}', f'shape{self.things[0].shape}'))
        return ('has-shapecolors', tuple((f'color{o.color}', f'shape{o.shape}') for o in self.things))

    def __repr__(self) -> str:
        res = self.__class__.__name__ + '(things=' + str(self.things) + ')'
        return res

    def __eq__(self, b: object) -> bool:
        if not isinstance(b, self.__class__):
            return False
        return self.things == b.things

    def create_grid_from_things(self, grid_size: int, things: Iterable[things_lib.ShapeColor]) -> grid_lib.Grid:
        grid = grid_lib.Grid(size=grid_size)
        for shapecolor in things:
            pos = grid.generate_available_pos(r=self.r)
            grid.add_object(pos=pos, o=shapecolor)
        return grid

    def create_positive_example(self, grid_size: int, num_distractors: int):
        distractors: List[things_lib.ShapeColor] = []
        tries = 0
        while len(distractors) < num_distractors:
            distractor = self.distractor_thing_space.sample()
            if distractor not in self.things:
                distractors.append(distractor)
            tries += 1
            if tries > 5000:
                print('self.things', self.things)
                raise spaces.SpaceNoValuesAvailable()
        ex_things = self.things + distractors
        return self.create_grid_from_things(grid_size=grid_size, things=ex_things)

    def create_negative_example(self, grid_size: int, num_distractors: int):
        new_things = list(self.things)
        neg_types = ['color', 'shape']

        def swap_color(old_o: things_lib.ShapeColor) -> things_lib.ShapeColor:
            new_o = old_o
            tries = 0
            while new_o == old_o or new_o not in self.distractor_thing_space:
                new_color = self.thing_space.color_space.sample()
                new_o = things_lib.ShapeColor(color=new_color, shape=old_o.shape)
                tries += 1
                if tries > 50:
                    raise spaces.SpaceNoValuesAvailable()
            return new_o

        def swap_shape(old_o: things_lib.ShapeColor) -> things_lib.ShapeColor:
            new_o = old_o
            tries = 0
            while new_o == old_o or new_o not in self.distractor_thing_space:
                new_shape = self.thing_space.shape_space.sample()
                new_o = things_lib.ShapeColor(color=old_o.color, shape=new_shape)
                tries += 1
                if tries > 50:
                    raise spaces.SpaceNoValuesAvailable()
            return new_o

        tries = 0
        while True:
            neg_type = self.r.choice(neg_types)
            corrupt_idx = self.r.randint(len(self.things))
            old_o = self.things[corrupt_idx]
            try:
                if neg_type == 'color':
                    new_o = swap_color(old_o=old_o)
                elif neg_type == 'shape':
                    new_o = swap_shape(old_o=old_o)
                new_things[corrupt_idx] = new_o
                break
            except spaces.SpaceNoValuesAvailable:
                pass
            if tries >= 100:
                print('self.things', self.things)
                raise spaces.SpaceNoValuesAvailable()

        distractors: List[things_lib.ShapeColor] = []
        tries = 0
        while len(distractors) < num_distractors:
            distractor = self.distractor_thing_space.sample()
            if distractor not in self.things:
                distractors.append(distractor)
            tries += 1
            if tries > 5000:
                print('self.things', self.things)
                raise spaces.SpaceNoValuesAvailable()
        ex_things = new_things + distractors
        return self.create_grid_from_things(grid_size=grid_size, things=ex_things)

    def as_seq(self) -> Tuple[torch.Tensor, List[str]]:
        raise NotImplementedError()


class ColorsHypothesis(Hypothesis):
    def __init__(self, multi_int_unordered_hyp: MultiIntUnorderedHypothesis, shape_space: spaces.IntSpace):
        self.multi_int_unordered_hyp = multi_int_unordered_hyp
        self.r = multi_int_unordered_hyp.r
        self.colors = multi_int_unordered_hyp.ints
        self.shape_space = shape_space

    @classmethod
    def from_english(
            cls,
            r: np.random.RandomState,
            hyp_eng: str,
            num_colors: int,
            shape_space: spaces.IntSpace) -> 'ColorsHypothesis':
        words = hyp_eng.split()
        assert words[0] == 'has-colors'
        num_entities = len(words) - 1
        colors = []
        words = words[1:]
        for i in range(num_entities):
            color_word = words[i]
            assert color_word.startswith('color')
            color = int(color_word.replace('color', '', 1))
            colors.append(color)
        multi_int_unordered_hyp = MultiIntUnorderedHypothesis(
            r=r, ints=tuple(colors), dim_length=num_colors)
        return cls(multi_int_unordered_hyp=multi_int_unordered_hyp, shape_space=shape_space)

    def as_english(self) -> str:
        return 'has-colors ' + ' '.join([f'color{color}' for color in self.colors])

    def as_english_structure(self) -> Union[str, Tuple[Any, Any]]:
        if len(self.colors) == 1:
            return ('has-colors', f'color{self.colors[0]}')
        return ('has-colors', tuple(f'color{color}' for color in self.colors))

    def __repr__(self) -> str:
        things = [things_lib.ShapeColor(shape=0, color=color) for color in self.multi_int_unordered_hyp.ints]
        res = self.__class__.__name__ + '(colors=' + str(things) + ')'
        return res

    def __eq__(self, b: object) -> bool:
        if not isinstance(b, ColorsHypothesis):
            return False
        return self.multi_int_unordered_hyp == b.multi_int_unordered_hyp

    def create_grid_from_colors(self, grid_size: int, colors: Tuple[int, ...]) -> grid_lib.Grid:
        ex_shapes = [self.shape_space.sample() for i in range(len(colors))]
        shapecolors = [things_lib.ShapeColor(shape=shape, color=color) for shape, color in zip(ex_shapes, colors)]
        grid = grid_lib.Grid(size=grid_size)
        for shapecolor in shapecolors:
            pos = grid.generate_available_pos(r=self.r)
            grid.add_object(pos=pos, o=shapecolor)
        return grid

    def create_positive_example(self, grid_size: int, num_distractors: int):
        ex_colors = self.multi_int_unordered_hyp.create_positive_example(num_distractors=num_distractors)
        return self.create_grid_from_colors(grid_size=grid_size, colors=ex_colors)

    def create_negative_example(self, grid_size: int, num_distractors: int):
        ex_colors = self.multi_int_unordered_hyp.create_negative_example(num_distractors=num_distractors)
        return self.create_grid_from_colors(grid_size=grid_size, colors=ex_colors)

    def as_seq(self) -> Tuple[torch.Tensor, List[str]]:
        raise NotImplementedError()


class ShapesHypothesis(Hypothesis):
    def __init__(self, multi_int_unordered_hyp: MultiIntUnorderedHypothesis, color_space: spaces.IntSpace):
        self.multi_int_unordered_hyp = multi_int_unordered_hyp
        self.shapes = multi_int_unordered_hyp.ints
        self.color_space = color_space
        self.r = multi_int_unordered_hyp.r

    def as_english(self) -> str:
        return 'has-shapes ' + ' '.join([f'shape{shape}' for shape in self.shapes])

    def as_english_structure(self) -> Union[str, Tuple[Any, Any]]:
        if len(self.shapes) == 1:
            return ('has-shapes', f'shape{self.shapes[0]}')
        return ('has-shapes', tuple(f'shape{shape}' for shape in self.shapes))

    def __repr__(self) -> str:
        res = self.__class__.__name__ + '(shapes=' + str(self.multi_int_unordered_hyp.ints) + ')'
        return res

    def __eq__(self, b: object) -> bool:
        if not isinstance(b, ColorsHypothesis):
            return False
        return self.multi_int_unordered_hyp == b.multi_int_unordered_hyp

    def create_grid_from_shapes(self, grid_size: int, shapes: Tuple[int, ...]) -> grid_lib.Grid:
        ex_colors = [self.color_space.sample() for i in range(len(shapes))]
        shapecolors = [things_lib.ShapeColor(shape=shape, color=color) for shape, color in zip(shapes, ex_colors)]
        grid = grid_lib.Grid(size=grid_size)
        for shapecolor in shapecolors:
            pos = grid.generate_available_pos(r=self.r)
            grid.add_object(pos=pos, o=shapecolor)
        return grid

    def create_positive_example(self, grid_size: int, num_distractors: int):
        ex_shapes = self.multi_int_unordered_hyp.create_positive_example(num_distractors=num_distractors)
        return self.create_grid_from_shapes(grid_size=grid_size, shapes=ex_shapes)

    def create_negative_example(self, grid_size: int, num_distractors: int):
        ex_shapes = self.multi_int_unordered_hyp.create_negative_example(num_distractors=num_distractors)
        return self.create_grid_from_shapes(grid_size=grid_size, shapes=ex_shapes)

    def as_seq(self) -> Tuple[torch.Tensor, List[str]]:
        raise NotImplementedError()
