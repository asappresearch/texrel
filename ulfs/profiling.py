import time
import torch
from collections import defaultdict


def synchronize():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


class Timer(object):
    """
    call 'state' *before* each thing you're interested in measuring, with the name of that thing

    in progress

    eg:

    timer.reset()
    do_a()
    timer.state('a')
    do_b()
    timer.state('b')
    timer.dump()
    """
    def __init__(self):
        self.reset()

    def reset(self):
        synchronize()
        self.times_by_state = defaultdict(float)
        self.last_time = time.time()
        self.last_state = None

    def state(self, state):
        synchronize()
        if self.last_state is not None:
            self.times_by_state[self.last_state] += (time.time() - self.last_time)
        self.last_time = time.time()
        self.last_state = state

    def dump(self):
        times = [{'name': name, 'time': time} for name, time in self.times_by_state.items()]
        times.sort(key=lambda x: x['time'], reverse=True)
        print('timings:')
        for t in times:
            print('    ', t['name'], '%.3f' % t['time'])
        self.reset()
