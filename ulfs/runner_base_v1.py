import time
import argparse
import inspect
import glob
import subprocess
from typing import Optional

import torch
import numpy as np

from ulfs import name_utils, utils, file_utils
from ulfs.params import Params
from ulfs.logger import Logger


def run(cmd_list, tail_lines=0):
    return '\n'.join(subprocess.check_output(cmd_list).decode('utf-8').split('\n')[- tail_lines:]).strip()


class RunnerBase(object):
    def __init__(
            self,
            additional_save_keys=[],
            save_as_statedict_keys=['model', 'opt'],
            step_key='epoch'
    ):
        utils.clean_argv()
        self.parser = argparse.ArgumentParser()
        self.params_l = []
        self.arguments_l = []
        self.logger = None
        self.additional_save_keys = additional_save_keys
        self.save_as_statedict_keys = save_as_statedict_keys
        self.step_key = step_key
        self.args_parsed = False
        self.setup_base_run = False
        self.init_run = True
        setattr(self, self.step_key, 0)

    def add_param(self, cmd, **kwargs):
        cmd_replaced = cmd.replace('-', '_')
        while cmd_replaced.startswith('_'):
            cmd_replaced = cmd_replaced[1:]
        # if cmd_replaced.startswith('--'):
        #     cmd_replaced = cmd_replaced[2:]
        # if cmd_replaced.startswith('-'):
        #     cmd_replaced = cmd_replaced[1:]
        # cmd_replaced = cmd_replaced.replace('-', '_')
        self.params_l.append(cmd_replaced)
        self.parser.add_argument(cmd, **kwargs)

    def add_argument(self, cmd, type, default=None, help=None):
        self.arguments_l.append(cmd.replace('--', '').replace('-', '_'))
        self.parser.add_argument(cmd, type=type, default=default, help=help)

    def parse_args(self, argv=None):
        assert not self.args_parsed
        self._add_standard_arguments(self.parser)
        self.args = self.parser.parse_args(argv)
        self._extract_standard_args(self.args)
        self._extract_params(self.args, self.params_l)
        self.args_parsed = True

    def setup(self, params):
        print('warning: you didnt override setup(self, params)')
        pass

    def start_mlflow(self):
        self.logger.start_mlflow()

    def setup_base(self, params=None, delay_mlflow: bool = False):
        assert 'init_run' in self.__dict__
        assert not self.setup_base_run
        if not self.args_parsed and not params:
            self.parse_args()
        if params:
            self.params = params
        file_utils.ensure_dirs_exist(['tmp', 'logs'])
        self.last_print = time.time()
        self.last_print_step = 0
        self.last_save = time.time()
        self.maybe_load_params()
        print('params', self.params)
        if self.logfile is not None and self.logfile != '':
            print('file', self._get_caller_filename(), 'delay_mlflow', delay_mlflow)
            self.logger = Logger(
                self.logfile, self.params, file=self._get_caller_filename(),
                delay_mlflow=delay_mlflow)
        if self.enable_cuda:
            # torch.use_deterministic_algorithms(True)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        self.setup(self.params)
        self.maybe_load_models()
        self.setup_base_run = True
        self.start_time = time.time()

    def run_base(self):
        if not self.setup_base_run:
            self.setup_base()
        self.finish = False
        while not self.finish:
            self.step_base()
        if self.logger is not None:
            self.logger.close()

    def incr_step(self):
        old_step = getattr(self, self.step_key)
        setattr(self, self.step_key, old_step + 1)

    def step_base(self):
        self.step(self.params)
        self.incr_step()
        self.maybe_save()

    def _get_caller_frame(self):
        stack = inspect.stack()
        d = 0
        while stack[d].filename == __file__:
            d += 1
        previous_frame = stack[d]
        return previous_frame

    def _get_caller_filename(self):
        previous_frame = self._get_caller_frame()
        # previous_frame = inspect.stack()[2]
        filename = previous_frame.filename
        return filename

    # def _maybe_add_argument(self, parser, name, *args, **kwargs):
    #     if name not in parser.option_string_actions:
    #         print('adding std arg', name)
    #         parser.add_argument(name, *args, **kwargs)
    #     else:
    #         print('info: ignoring existing std arg', name)

    def _add_standard_arguments(self, parser):
        # previous_frame = inspect.stack()[2]
        # filename = previous_frame.filename
        filename = self._get_caller_filename()
        file = filename.split('/')[-1].split('.')[0]
        print('file', file)

        # if '--disable-cuda' not in parser.option_string_actions:
        parser.add_argument('--disable-cuda', action='store_true')
        parser.add_argument('--save-every-seconds', type=int, default=-1)
        parser.add_argument('--render-every-seconds', type=int, default=30, help='-1 disables')
        parser.add_argument('--render-every-steps', type=int, default=-1, help='-1 disables')
        parser.add_argument('--name', type=str, default=file, help='used for logfile naming')
        parser.add_argument('--load-last-model', action='store_true')
        parser.add_argument('--model-file', type=str, default='tmp/{name}_{ref}_{hostname}_{date}_{time}.dat')
        parser.add_argument('--logfile', type=str, default='logs/log_{name}_{ref}_{hostname}_{date}_{time}.log')

        # self._maybe_add_argument(parser, '--disable-cuda', action='store_true')
        # self._maybe_add_argument(parser, '--save-every-seconds', type=int, default=300)
        # self._maybe_add_argument(parser, '--render-every-seconds', type=int, default=30)
        # self._maybe_add_argument(parser, '--name', type=str, default=file, help='used for logfile naming')
        # self._maybe_add_argument(parser, '--model-file',
        #     type=str, default='tmp/{name}_{ref}_{hostname}_{date}_{time}.dat')
        # self._maybe_add_argument(parser, '--logfile',
        #     type=str, default='logs/log_{name}_{ref}_{hostname}_{date}_{time}.log')

        self.add_param('--ref', type=str, required=True, help='your experiment reference')

    def _extract_standard_args(self, args):
        utils.reverse_args(args, 'disable_cuda', 'enable_cuda')
        self.params_l.append('enable_cuda')
        self.save_every_seconds = args.save_every_seconds
        self.render_every_seconds = args.render_every_seconds
        self.render_every_steps = args.render_every_steps
        self.load_last_model = args.load_last_model
        self.model_file_glob = args.model_file.format(
            **args.__dict__, hostname='*', date='*', time='*')
        self.model_file = args.model_file.format(
            **args.__dict__, hostname=name_utils.hostname(),
            date=name_utils.date_string(), time=name_utils.time_string())
        self.logfile = args.logfile.format(
            **args.__dict__, hostname=name_utils.hostname(),
            date=name_utils.date_string(), time=name_utils.time_string())
        self.standard_args_dict = {}
        for k in [
                'save_every_seconds', 'render_every_seconds', 'render_every_steps',
                'model_file', 'logfile', 'name', 'load_last_model']:
            self.standard_args_dict[k] = args.__dict__[k]
            del args.__dict__[k]

    def _extract_params(self, args, params_keys):
        self.params = Params()
        self.params.extract_from_args(args, params_keys=self.params_l)
        self.enable_cuda = self.params.enable_cuda
        self.torch_constr = torch.cuda if self.enable_cuda else torch

    def maybe_load_params(self):
        self.epoch = 0
        self.loaded_state = False
        self.start_time = time.time()
        if self.load_last_model:
            glob_str = self.model_file_glob
            # files = glob.glob(glob_str)
            files = glob.glob(glob_str)
            files.sort()
            file = files[-1]
            print('loading ', file)
            # files = run(['find', log_dir, '-cmin', f'-{time_filter_minutes}']).split('\n')
        # if path.isfile(self.model_file):
            with open(file, 'rb') as f:
                self.state = torch.load(f)
            params = self.state['params']
            self.loaded_state = True
            self.params = params
            self.p = params
            self.start_time = time.time() - self.state['elapsed_time']
        return self.params

    def load_models(self):
        for k in self.save_as_statedict_keys:
            getattr(self, k).load_state_dict(self.state[f'{k}_state'])
            print('loaded state dict key', k)
        setattr(self, self.step_key, self.state[self.step_key])
        for k in self.additional_save_keys:
            setattr(self, k, self.state[k])
            print('loaded value ', k, getattr(self, k))
        if 'torch_rng_state' in self.state:
            torch.set_rng_state(self.state['torch_rng_state'])
            print('loaded torch rng state')
        if 'numpy_rng_state' in self.state:
            np.random.set_state(self.state['numpy_rng_state'])
            print('loaded numpy rng state')
        print('loaded model,', self.step_key, getattr(self, self.step_key))

    def maybe_load_models(self):
        if self.loaded_state:
            self.load_models()

    def save_to(self, model_file):
        state = {}
        state['params'] = self.params
        for k in self.save_as_statedict_keys:
            state[f'{k}_state'] = getattr(self, k).state_dict()
        state[self.step_key] = getattr(self, self.step_key)
        state['elapsed_time'] = time.time() - self.start_time
        state['torch_rng_state'] = torch.get_rng_state()
        state['numpy_rng_state'] = np.random.get_state()
        for k in self.additional_save_keys:
            state[k] = getattr(self, k)
            print('saved ', k, state[k])
        file_utils.safe_save(model_file, state)

    def save(self):
        if self.model_file is None or self.model_file == '':
            return
        self.save_to(self.model_file)
        self.last_save = time.time()

    def should_render(self):
        if self.render_every_seconds > 0 and time.time() - self.last_print >= self.render_every_seconds:
            return True
        if (
            self.render_every_steps > 0 and
            getattr(self, self.step_key) - self.last_print_step >= self.render_every_steps
        ):
            return True
        return False

    def maybe_save(self):
        if self.save_every_seconds <= 0:
            return
        if time.time() - self.last_save >= self.save_every_seconds:
            self.save()

    def apply_loss(self, loss):
        self.opt.zero_grad()
        loss.backward()
        self.opt.step()

    def iter_epochs(self):
        while True:
            yield self.epoch
            setattr(self, self.step_key, getattr(self, self.step_key) + 1)

    def log(self, logdict, step: int = 0):
        if self.logfile is not None and self.logfile != '':
            self.logger.log(logdict, step=step)

    def maybe_print_and_log(self, stats_dict, formatstr, step: int = 0):
        if self.should_render():
            self.print_and_log(stats_dict, formatstr, step=step)

    def print_and_log(self, stats_dict, formatstr, step: Optional[int] = None):
        stats_dict[self.step_key] = getattr(self, self.step_key)
        stats_dict['elapsed_time'] = time.time() - self.start_time
        print(formatstr.format(**stats_dict))
        if step is None:
            step = getattr(self, self.step_key)
        if self.logfile is not None and self.logfile != '':
            self.log(stats_dict, step=step)
        self.last_print = time.time()
        self.last_print_step = getattr(self, self.step_key)
