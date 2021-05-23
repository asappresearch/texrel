import json
from os import path
import os
import time
import datetime


class Logger(object):
    def __init__(self, flush_every_seconds=10):
        self.flush_every_seconds = flush_every_seconds

    def add_to_parser(self, parser):
        parser.add_argument('--logfile', type=str, default='logs/{name}%Y%m%d_%H%M%S.log')

    def eat_args(self, name, args):
        self.logfile = datetime.datetime.strftime(datetime.datetime.now(), args.logfile)
        self.name = name
        if self.name is not None and self.name != '':
            self.logfile = self.logfile.format(
                name=self.name + '_')
        else:
            self.logfile = self.logfile.format(
                name='')
        del args.__dict__['logfile']
        if not path.isdir(path.dirname(self.logfile)):
            os.makedirs(path.dirname(self.logfile))
        self.f = open(self.logfile, 'a')
        self.last_flush = time.time()

    def log(self, datadict):
        self.f.write(json.dumps(datadict) + '\n')
        if time.time() - self.last_flush >= self.flush_every_seconds:
            self.f.flush()
            self.last_flush = time.time()

    def log_dicts(self, dicts):
        alldict = {}
        for name, adict in dicts.items():
            for k, v in adict.items():
                alldict[name + k] = v
        try:
            self.f.write(json.dumps(alldict) + '\n')
        except Exception as e:
            print(e)
            for k, v in alldict.items():
                print(k, type(v), v)
            raise e
        if time.time() - self.last_flush >= self.flush_every_seconds:
            self.f.flush()
            self.last_flush = time.time()
