"""
create training examples, and test distractors, for various relations

we need to do a few things:
- come up with a way to generate relations
- (a way to test relations potentially)
- come up with a way to convert relations into examples
- come up with a way to convert relations into distrators
  (maybe change one thing? like, a color, a position, something like that?)
"""


class Dataset(object):
    def __init__(self, rel_space, grid_size):
        self.rel_space = rel_space
        self.thing_space = rel_space.thing_space
        self.grid_size = grid_size

    def get_grid_planes(self):
        thing_space = self.thing_space
        return thing_space.color_space.size + \
            thing_space.shape_space.size
