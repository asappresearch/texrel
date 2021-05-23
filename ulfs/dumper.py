"""
Dump a bunch of state when some weird condition is detected, such as zero weights etc
"""
import os
import torch
from os import path
from os.path import join
import datetime


class Dumper(object):
    def __init__(self, model=None, opt=None, dump_dir='debug_dumps'):
        """
        we'll call .state_dict() on each of model and opt, and dump those too
        """
        self.dump_dir = dump_dir
        if not path.isdir(dump_dir):
            os.makedirs(dump_dir)
        self.model = model
        self.opt = opt

    def dump(self, object_list):
        filename = datetime.datetime.strftime(datetime.datetime.now(), 'dump_%Y%m%d_%H%M%S.dat')
        dump_path = join(self.dump_dir, filename)
        cpu_objects = [o.cpu() for o in object_list if hasattr(o, 'cpu')]
        save_dict = {}
        save_dict['objects'] = object_list
        save_dict['cpu_objects'] = cpu_objects
        if self.model is not None:
            save_dict['model_state'] = self.model.state_dict()
        if self.opt is not None:
            save_dict['opt_state'] = self.opt.state_dict()
        with open(dump_path, 'wb') as f:
            torch.save(save_dict, f)
        print('dumped objects to', dump_path)
