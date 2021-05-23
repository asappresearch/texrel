import time
from collections import defaultdict


class Timer(object):
    def __init__(self):
        self.last = time.time()

    def lap(self, msg=''):
        elapsed = time.time() - self.last
        print(msg, elapsed)
        self.last = time.time()


class StatefulTimer(object):
    """
    Adapted from https://github.com/hughperkins/EasyCL/blob/master/util/StatefulTimer.cpp and
    https://github.com/hughperkins/EasyCL/blob/master/util/StatefulTimer.h
    """
    def __init__(self):
        self.reset()

    def state(self, state):
        if self.last_state is not None:
            elapsed = time.time() - self.last_time
            self.secs_by_state[self.last_state] += elapsed
        self.last_state = state
        self.last_time = time.time()

    def dump(self):
        for state, secs in self.secs_by_state.items():
            print(state, '%.3f' % secs)

    def reset(self):
        self.last_time = time.time()
        self.last_state = None
        self.secs_by_state = defaultdict(float)
