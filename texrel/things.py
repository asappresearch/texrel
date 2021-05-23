"""
a 'thing' is a physical object. I'd prefer to say 'object', but that gets confused with
python / oop objects
"""
import functools
# import random
from typing import List, Tuple, Union, Any, TYPE_CHECKING

import torch
# import numpy as np

from colorama import Fore

if TYPE_CHECKING:
    from texrel import spaces


# this is used for displaying objects as short strings
_colors = [
    Fore.RED, Fore.YELLOW, Fore.BLUE, Fore.GREEN, Fore.CYAN, Fore.MAGENTA,
    Fore.BLACK, Fore.LIGHTRED_EX, Fore.LIGHTGREEN_EX,
    Fore.LIGHTBLACK_EX, Fore.LIGHTBLUE_EX, Fore.LIGHTCYAN_EX, Fore.LIGHTMAGENTA_EX,
    Fore.LIGHTWHITE_EX, Fore.LIGHTYELLOW_EX, Fore.BLACK
]


# this is used for displaying objects as short strings
_shapes = '#O^U@XVC$ABCDEFG'


@functools.total_ordering
class ShapeColor(object):
    def __init__(self, shape: int, color: int) -> None:
        self.shape = shape
        self.color = color

    def as_english(self) -> str:
        """
        eg "color14 shape4"
        zero-indexed (for consistency)
        """
        return f'color{self.color} shape{self.shape}'

    def as_english_structure(self) -> Union[str, Tuple[Any, Any]]:
        """
        returns a tuple of shape string and color string
        """
        return (f'color{self.color}', f'shape{self.shape}')

    def __repr__(self) -> str:
        return _colors[self.color] + _shapes[self.shape] + Fore.RESET

    def __eq__(self, second: object) -> bool:
        if not isinstance(second, ShapeColor):
            return False
        return self.shape == second.shape and self.color == second.color

    def __lt__(self, second: object) -> bool:
        if not isinstance(second, ShapeColor):
            return False
        if self.shape != second.shape:
            return self.shape < second.shape
        return self.color < second.color

    def __hash__(self) -> int:
        return self.shape * len(_colors) + self.color

    def as_onehot_indices(self, thing_space: 'spaces.ThingSpace') -> List[int]:
        return [
            self.shape,
            self.color + thing_space.shape_space.onehot_size
        ]

    def as_indices(self) -> Tuple[List[int], List[str]]:
        return [
            self.shape,
            self.color
        ], ['S', 'C']

    def as_onehot_tensor_size(self, thing_space: 'spaces.ThingSpace') -> int:
        return thing_space.shape_space.onehot_size + thing_space.color_space.onehot_size

    @classmethod
    def eat_from_indices(cls, indices: List[int]) -> Tuple['ShapeColor', List[int]]:
        shape, color = indices[:2]
        indices = indices[2:]
        return ShapeColor(shape=shape, color=color), indices

    @classmethod
    def from_onehot_tensor(cls, thing_space: 'spaces.ThingSpace', tensor: torch.Tensor) -> 'ShapeColor':
        return cls.eat_from_onehot_tensor(thing_space, tensor)[0]

    @classmethod
    def eat_from_onehot_tensor(cls, thing_space: 'spaces.ThingSpace', tensor: torch.Tensor) -> Tuple[
            'ShapeColor', torch.Tensor]:
        """
        returns tensor with shapcecolor removed from front
        """
        color_space = thing_space.color_space
        shape_space = thing_space.shape_space

        def eat(tensor, size):
            id = tensor.view(-1)[:size].nonzero().view(-1)[0]
            tensor = tensor[size:]
            return id, tensor

        shape, tensor = eat(tensor=tensor, size=shape_space.onehot_size)
        color, tensor = eat(tensor=tensor, size=color_space.onehot_size)
        return ShapeColor(shape=shape, color=color), tensor
