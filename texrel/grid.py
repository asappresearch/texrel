from typing import Tuple, List, Optional

import torch
import numpy as np

from colorama import Fore

from texrel import things


class Grid(object):
    """
    first coordinate is row, top to bottom; second is column, left to right
    """
    def __init__(self, size):
        self.size = size
        self.grid: List[List[Optional[things.ShapeColor]]] = []
        for i in range(size):
            row = []
            for j in range(size):
                row.append(None)
            self.grid.append(row)
        self.objects_set = []

    def __eq__(self, b: object) -> bool:
        if not isinstance(b, self.__class__):
            return False
        return self.grid == b.grid

    def add_object(self, pos: Tuple[int, int], o: things.ShapeColor):
        assert self.grid[pos[0]][pos[1]] is None
        self.grid[pos[0]][pos[1]] = o
        self.objects_set.append(o)
        return self

    def get_pos_for_object(self, o: things.ShapeColor) -> Tuple[int, int]:
        """
        warning: slow

        first coordinate is y (ie vert), second coordinate is x (ie horiz)
        (this is mostly historical, because of how __repr__ function works,
        not sure I agree with this in hindsight...)
        """
        for i in range(self.size):
            for j in range(self.size):
                if self.grid[i][j] == o:
                    return (i, j)
        raise ValueError()

    def as_shape_color_tensors(self) -> Tuple[torch.Tensor, torch.Tensor]:
        shapes = torch.zeros((self.size, self.size), dtype=torch.int64)
        colors = torch.zeros((self.size, self.size), dtype=torch.int64)
        for i in range(self.size):
            for j in range(self.size):
                o: Optional[things.ShapeColor] = self.grid[i][j]
                if o is None:
                    continue
                shapes[i, j] = o.shape + 1
                colors[i, j] = o.color + 1
        return shapes, colors

    def __repr__(self) -> str:
        res_l = []
        for i in range(self.size):
            row = ''
            for j in range(self.size):
                o = self.grid[i][j]
                if o is None:
                    row += '.'
                else:
                    fore_color = things._colors[o.color]
                    row += fore_color
                    row += things._shapes[o.shape]
                    row += Fore.RESET
            res_l.append(row)
        return '\n'.join(res_l)

    def render(self) -> None:
        print(str(self))

    def generate_available_pos(self, r: np.random.RandomState) -> Tuple[int, int]:
        """
        returns a pos which is None at that position in the grid
        """
        pos = None
        while pos is None or self.grid[pos[0]][pos[1]] is not None:
            pos = r.choice(self.size, 2, replace=True)
        pos = tuple(pos)
        return pos
