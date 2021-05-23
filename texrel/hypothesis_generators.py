import abc
from typing import List
import math

import numpy as np

from texrel import hypothesis, spaces, things as things_lib


def _get_num_holdout_for_shapes_or_colors(
        num_entities: int,
        holdout_fraction: float,
        num_avail_shapes_or_colors: int,
        num_distractors: int) -> int:
    total_combos = (
        math.factorial(num_avail_shapes_or_colors + num_entities - 1) //
        math.factorial(num_entities) // math.factorial(num_avail_shapes_or_colors - 1))
    print('total_combos', total_combos)
    num_holdout = math.ceil(total_combos * holdout_fraction)
    print('num_holdout by frac', num_holdout)
    min_holdout = 1
    print('min_holdout', min_holdout)
    # max_holdout = total_combos - num_entities - num_distractors
    max_holdout = total_combos - num_entities
    print('max_holdout', max_holdout)
    if max_holdout < min_holdout:
        return -1
    num_holdout = max(min_holdout, num_holdout)
    num_holdout = min(max_holdout, num_holdout)

    if num_entities + num_distractors > num_avail_shapes_or_colors:
        return -1

    # num_holdout = max(num_holdout, num_entities)
    # print('num_holdout after max', num_holdout)
    # if total_combos - num_holdout < num_entities:
    #     return -1
    return num_holdout


class HypothesisGenerator(abc.ABC):
    @abc.abstractmethod
    def __call__(self):
        """
        generates a single RelationHypothesis
        """
        pass


class HypothesisGenerator2(abc.ABC):
    @abc.abstractmethod
    def sample_hyp(self, train: bool) -> hypothesis.Hypothesis:
        pass

    @abc.abstractclassmethod
    def get_num_holdout(
            cls, num_distractors: int,
            holdout_fraction: float,
            num_avail_colors: int,
            num_avail_shapes: int) -> int:
        pass


class RelationsHG(HypothesisGenerator2):
    def __init__(
            self, r: np.random.RandomState, available_preps: List[str],
            num_avail_colors: int, num_avail_shapes: int, num_holdout: int):
        self.r = r

        self.whole_thing_space = spaces.ThingSpace(
            r=r,
            color_space=spaces.IntSpace(r=r, num_ints=num_avail_colors),
            shape_space=spaces.IntSpace(r=r, num_ints=num_avail_shapes)
        )

        self.prep_space = spaces.PrepositionSpace(
            available_preps=available_preps, r=r)

        self.thing_space_training, self.thing_space_holdout = self.whole_thing_space.partition(
            partition_sizes=[len(self.whole_thing_space) - num_holdout, num_holdout]
        )

        self.rel_space_training = spaces.RelationSpace(
            prep_space=self.prep_space,
            thing_space=self.thing_space_training)
        self.rel_space_holdout = spaces.RelationSpace(
            prep_space=self.prep_space,
            thing_space=self.thing_space_holdout)

    @classmethod
    def get_num_holdout(
            cls, num_distractors: int, holdout_fraction: float, num_avail_colors: int, num_avail_shapes: int) -> int:
        return math.ceil(holdout_fraction * num_avail_colors * num_avail_shapes)

    def sample_hyp(self, train: bool):
        """
        generates a single RelationHypothesis
        """
        rel_space = self.rel_space_training if train else self.rel_space_holdout
        distractor_thing_space = self.thing_space_training if train else self.whole_thing_space
        relation = rel_space.sample()
        return hypothesis.RelationHypothesis(
            r=self.r,
            relation=relation,
            rel_space=rel_space,
            distractor_thing_space=distractor_thing_space
        )


class ThingsHG(HypothesisGenerator2):
    def __init__(
            self,
            r: np.random.RandomState,
            num_entities: int,
            num_holdout: int,
            num_avail_colors: int,
            num_avail_shapes: int):
        """
        hypothesis generator for hypotheses with one or more specific shapecolors in

        Parameters
        ----------
        r: np.random.RandomState
        num_entities: int
            how many shapecolors we are going to pick for each hypothesis
        num_avail_colors: int
            total number available colors
        num_avail_shapes: int
            total number available shapes
        """
        self.num_entities = num_entities
        self.num_holdout = num_holdout
        self.num_avail_colors = num_avail_colors
        self.num_avail_shapes = num_avail_shapes
        self.r = r

        self.whole_thing_space = spaces.ThingSpace(
            r=r,
            color_space=spaces.IntSpace(r=r, num_ints=num_avail_colors),
            shape_space=spaces.IntSpace(r=r, num_ints=num_avail_shapes)
        )

        self.thing_space_training, self.thing_space_holdout = self.whole_thing_space.partition(
            partition_sizes=[len(self.whole_thing_space) - num_holdout, num_holdout]
        )

    def sample_hyp(self, train: bool) -> hypothesis.Hypothesis:
        thing_space = self.thing_space_training if train else self.thing_space_holdout
        distractor_thing_space = self.thing_space_training if train else self.whole_thing_space
        entities: List[things_lib.ShapeColor] = []
        while len(entities) < self.num_entities:
            thing = thing_space.sample()
            entities.append(thing)
        return hypothesis.ThingsHypothesis(
            r=self.r,
            things=entities,
            thing_space=thing_space,
            distractor_thing_space=distractor_thing_space
        )

    def __repr__(self):
        return self.__class__.__name__ + '()'

    @classmethod
    def get_num_holdout(
            cls, num_distractors: int, holdout_fraction: float, num_avail_colors: int, num_avail_shapes: int) -> int:
        raise NotImplementedError()


class Things1HG(ThingsHG):
    def __init__(
            self, r: np.random.RandomState, num_holdout: int,
            num_avail_colors: int, num_avail_shapes: int):
        super().__init__(
            num_entities=1, r=r, num_holdout=num_holdout, num_avail_colors=num_avail_colors,
            num_avail_shapes=num_avail_shapes)

    @classmethod
    def get_num_holdout(
            cls, num_distractors: int, holdout_fraction: float, num_avail_colors: int, num_avail_shapes: int) -> int:
        return math.ceil(num_avail_shapes * num_avail_colors * holdout_fraction)


class Things2HG(ThingsHG):
    def __init__(
            self, r: np.random.RandomState, num_holdout: int,
            num_avail_colors: int, num_avail_shapes: int):
        super().__init__(
            num_entities=2, r=r, num_holdout=num_holdout, num_avail_colors=num_avail_colors,
            num_avail_shapes=num_avail_shapes)

    @classmethod
    def get_num_holdout(
            cls, num_distractors: int, holdout_fraction: float, num_avail_colors: int, num_avail_shapes: int) -> int:
        return math.ceil(num_avail_shapes * num_avail_colors * holdout_fraction)


class Things3HG(ThingsHG):
    def __init__(
            self, r: np.random.RandomState, num_holdout: int,
            num_avail_colors: int, num_avail_shapes: int):
        super().__init__(
            num_entities=3, r=r, num_holdout=num_holdout, num_avail_colors=num_avail_colors,
            num_avail_shapes=num_avail_shapes)

    @classmethod
    def get_num_holdout(
            cls, num_distractors: int, holdout_fraction: float, num_avail_colors: int, num_avail_shapes: int) -> int:
        num_entities = 3
        total_combos = num_avail_colors * num_avail_shapes
        num_holdout = math.ceil(total_combos * holdout_fraction)
        num_holdout = max(num_holdout, num_entities)
        if total_combos - num_holdout < num_entities:
            return -1
        return num_holdout


class MultiIntUnorderedGenerator(HypothesisGenerator2):
    def __init__(
        self,
        r: np.random.RandomState,
        num_dims: int,
        dim_length: int,
        num_holdout: int
    ):
        self.r = r
        self.num_dims = num_dims
        self.dim_length = dim_length
        self.num_holdout = num_holdout

        self.full_space = spaces.MultiIntUnorderedSpace(r=r, num_dims=num_dims, num_ints=dim_length)
        self.train_space, self.holdout_space = self.full_space.partition(
            sizes=[len(self.full_space) - num_holdout, num_holdout])
        assert len(self.train_space) >= len(self.holdout_space)

    def sample_hyp(self, train: bool):
        if train:
            ints = self.train_space.sample()
            distractor_space = self.train_space
        else:
            ints = self.holdout_space.sample()
            distractor_space = self.full_space
        return hypothesis.MultiIntUnorderedHypothesis(
            r=self.r, ints=ints, dim_length=distractor_space.num_ints)

    def get_num_holdout(
            cls, num_distractors: int, holdout_fraction: float, num_avail_colors: int, num_avail_shapes: int) -> int:
        raise NotImplementedError()


class ColorsHG(HypothesisGenerator2):
    def __init__(
            self,
            r: np.random.RandomState,
            num_entities: int,
            num_holdout: int,
            num_avail_colors: int,
            num_avail_shapes: int):
        """
        hypothesis generator for hypotheses with one or more specific colors in

        Parameters
        ----------
        r: np.random.RandomState
        num_entities: int
            how many colors we are going to pick for each hypothesis
        num_avail_colors: int
            total number available colors
        num_avail_shapes: int
            total number available shapes
        """
        self.num_entities = num_entities
        self.num_holdout = num_holdout
        self.num_avail_colors = num_avail_colors
        self.num_avail_shapes = num_avail_shapes
        self.r = r

        self.shape_space = spaces.IntSpace(r=self.r, num_ints=num_avail_shapes)
        self.multi_int_unordered_generator = MultiIntUnorderedGenerator(
            r=r, num_dims=num_entities, num_holdout=num_holdout, dim_length=num_avail_colors)

    def sample_hyp(self, train: bool) -> hypothesis.Hypothesis:
        multi_int_unordered_hyp = self.multi_int_unordered_generator.sample_hyp(train=train)
        colors_hyp = hypothesis.ColorsHypothesis(
            multi_int_unordered_hyp=multi_int_unordered_hyp, shape_space=self.shape_space)
        return colors_hyp

    def __repr__(self):
        return self.__class__.__name__ + '()'

    @classmethod
    def get_num_holdout(
            cls, num_distractors: int, holdout_fraction: float, num_avail_colors: int, num_avail_shapes: int) -> int:
        raise NotImplementedError()

    @classmethod
    def _get_num_holdout(
            cls,
            num_entities: int,
            holdout_fraction: float,
            num_avail_colors: int,
            num_avail_shapes: int,
            num_distractors: int) -> int:
        return _get_num_holdout_for_shapes_or_colors(
            num_entities=num_entities,
            holdout_fraction=holdout_fraction,
            num_avail_shapes_or_colors=num_avail_colors,
            num_distractors=num_distractors
        )
        # total_combos = (
        #     math.factorial(num_avail_colors + num_entities - 1) //
        #     math.factorial(num_entities) // math.factorial(num_avail_colors - 1))
        # print('total_combos', total_combos)
        # num_holdout = math.ceil(total_combos * holdout_fraction)
        # print('num_holdout by frac', num_holdout)
        # min_holdout = 1
        # print('min_holdout', min_holdout)
        # max_holdout = total_combos - num_entities - num_distractors
        # print('max_holdout', max_holdout)
        # if max_holdout < min_holdout:
        #     return -1
        # num_holdout = max(min_holdout, num_holdout)
        # num_holdout = min(max_holdout, num_holdout)
        # # num_holdout = max(num_holdout, num_entities)
        # # print('num_holdout after max', num_holdout)
        # # if total_combos - num_holdout < num_entities:
        # #     return -1
        # return num_holdout


class Colors1HG(ColorsHG):
    def __init__(
            self, r: np.random.RandomState, num_holdout: int,
            num_avail_colors: int, num_avail_shapes: int):
        super().__init__(
            num_entities=1, r=r, num_holdout=num_holdout, num_avail_colors=num_avail_colors,
            num_avail_shapes=num_avail_shapes)

    @classmethod
    def get_num_holdout(
            cls, num_distractors: int, holdout_fraction: float, num_avail_colors: int, num_avail_shapes: int) -> int:
        return cls._get_num_holdout(
            holdout_fraction=holdout_fraction,
            num_avail_colors=num_avail_colors,
            num_avail_shapes=num_avail_shapes,
            num_distractors=num_distractors,
            num_entities=1
        )


class Colors2HG(ColorsHG):
    def __init__(
            self, r: np.random.RandomState, num_holdout: int,
            num_avail_colors: int, num_avail_shapes: int):
        super().__init__(
            num_entities=2, r=r, num_holdout=num_holdout, num_avail_colors=num_avail_colors,
            num_avail_shapes=num_avail_shapes)

    @classmethod
    def get_num_holdout(
            cls, num_distractors: int, holdout_fraction: float, num_avail_colors: int, num_avail_shapes: int) -> int:
        return cls._get_num_holdout(
            holdout_fraction=holdout_fraction,
            num_avail_colors=num_avail_colors,
            num_avail_shapes=num_avail_shapes,
            num_distractors=num_distractors,
            num_entities=2
        )


class Colors3HG(ColorsHG):
    def __init__(
            self, r: np.random.RandomState, num_holdout: int,
            num_avail_colors: int, num_avail_shapes: int):
        super().__init__(
            num_entities=3, r=r, num_holdout=num_holdout, num_avail_colors=num_avail_colors,
            num_avail_shapes=num_avail_shapes)

    @classmethod
    def get_num_holdout(
            cls, num_distractors: int, holdout_fraction: float, num_avail_colors: int, num_avail_shapes: int) -> int:
        return cls._get_num_holdout(
            holdout_fraction=holdout_fraction,
            num_avail_colors=num_avail_colors,
            num_avail_shapes=num_avail_shapes,
            num_distractors=num_distractors,
            num_entities=3
        )


class ShapesHG(HypothesisGenerator2):
    def __init__(
            self,
            r: np.random.RandomState,
            num_entities: int,
            num_holdout: int,
            num_avail_colors: int,
            num_avail_shapes: int):
        """
        hypothesis generator for hypotheses with one or more specific shapes in

        Parameters
        ----------
        r: np.random.RandomState
        num_entities: int
            how many colors we are going to pick for each hypothesis
        num_avail_colors: int
            total number available colors
        num_avail_shapes: int
            total number available shapes
        """
        self.num_entities = num_entities
        self.num_holdout = num_holdout
        self.num_avail_colors = num_avail_colors
        self.num_avail_shapes = num_avail_shapes
        self.r = r

        self.color_space = spaces.IntSpace(r=self.r, num_ints=num_avail_colors)
        self.multi_int_unordered_generator = MultiIntUnorderedGenerator(
            r=r, num_dims=num_entities, num_holdout=num_holdout, dim_length=num_avail_shapes)

    def sample_hyp(self, train: bool) -> hypothesis.Hypothesis:
        multi_int_unordered_hyp = self.multi_int_unordered_generator.sample_hyp(train=train)
        shapes_hyp = hypothesis.ShapesHypothesis(
            multi_int_unordered_hyp=multi_int_unordered_hyp, color_space=self.color_space)
        return shapes_hyp

    def __repr__(self):
        return self.__class__.__name__ + '()'

    @classmethod
    def get_num_holdout(
            cls,
            num_distractors: int,
            holdout_fraction: float,
            num_avail_colors: int,
            num_avail_shapes: int) -> int:
        raise NotImplementedError()

    @classmethod
    def _get_num_holdout(
            cls,
            num_distractors: int,
            num_entities: int,
            holdout_fraction: float,
            num_avail_colors: int,
            num_avail_shapes: int) -> int:
        return _get_num_holdout_for_shapes_or_colors(
            num_entities=num_entities,
            holdout_fraction=holdout_fraction,
            num_avail_shapes_or_colors=num_avail_shapes,
            num_distractors=num_distractors
        )
        # total_combos = (
        #     math.factorial(num_avail_shapes + num_entities - 1) //
        #     math.factorial(num_entities) // math.factorial(num_avail_shapes - 1))
        # print('total_combos', total_combos)
        # num_holdout = math.ceil(total_combos * holdout_fraction)
        # print('num_holdout by frac', num_holdout)
        # num_holdout = max(num_holdout, num_entities)
        # print('num_holdout after max', num_holdout)
        # if total_combos - num_holdout < num_entities:
        #     return -1
        # return num_holdout


class Shapes1HG(ShapesHG):
    def __init__(
            self, r: np.random.RandomState, num_holdout: int,
            num_avail_colors: int, num_avail_shapes: int):
        super().__init__(
            num_entities=1, r=r, num_holdout=num_holdout, num_avail_colors=num_avail_colors,
            num_avail_shapes=num_avail_shapes)

    @classmethod
    def get_num_holdout(
            cls, num_distractors: int, holdout_fraction: float, num_avail_colors: int, num_avail_shapes: int) -> int:
        return cls._get_num_holdout(
            num_distractors=num_distractors,
            holdout_fraction=holdout_fraction,
            num_avail_colors=num_avail_colors,
            num_avail_shapes=num_avail_shapes,
            num_entities=1
        )


class Shapes2HG(ShapesHG):
    def __init__(
            self, r: np.random.RandomState, num_holdout: int,
            num_avail_colors: int, num_avail_shapes: int):
        super().__init__(
            num_entities=2, r=r, num_holdout=num_holdout, num_avail_colors=num_avail_colors,
            num_avail_shapes=num_avail_shapes)

    @classmethod
    def get_num_holdout(
            cls, num_distractors: int, holdout_fraction: float, num_avail_colors: int, num_avail_shapes: int) -> int:
        return cls._get_num_holdout(
            num_distractors=num_distractors,
            holdout_fraction=holdout_fraction,
            num_avail_colors=num_avail_colors,
            num_avail_shapes=num_avail_shapes,
            num_entities=2
        )


class Shapes3HG(ShapesHG):
    def __init__(
            self, r: np.random.RandomState, num_holdout: int,
            num_avail_colors: int, num_avail_shapes: int):
        super().__init__(
            num_entities=3, r=r, num_holdout=num_holdout, num_avail_colors=num_avail_colors,
            num_avail_shapes=num_avail_shapes)

    @classmethod
    def get_num_holdout(
            cls, num_distractors: int, holdout_fraction: float, num_avail_colors: int, num_avail_shapes: int) -> int:
        return cls._get_num_holdout(
            num_distractors=num_distractors,
            holdout_fraction=holdout_fraction,
            num_avail_colors=num_avail_colors,
            num_avail_shapes=num_avail_shapes,
            num_entities=3
        )
