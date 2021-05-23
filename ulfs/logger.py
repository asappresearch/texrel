import os
from typing import Dict, Any
from os import path
import json
import sys
import datetime
import mlflow

from ulfs import git_info


class Logger(object):
    def __init__(self, logfile, params, file, delay_mlflow):
        self.use_mlflow = 'MLFLOW_TRACKING_URI' in os.environ
        self.mlflow_started = False
        if not path.isdir('logs'):
            os.makedirs('logs')
        self.f = open(logfile, 'w')
        self.ref = params.ref
        meta = {}
        meta['params'] = params.__dict__
        meta['file'] = path.splitext(path.basename(file))[0]
        meta['argv'] = sys.argv
        meta['hostname'] = os.uname().nodename
        meta['gitlog'] = git_info.get_git_log()
        meta['gitdiff'] = git_info.get_git_diff()
        meta['start_datetime'] = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        self.meta = meta
        self.f.write(json.dumps(meta) + '\n')
        self.f.flush()
        print('logger, delay_mlflow', delay_mlflow, 'ref', self.ref)
        if self.use_mlflow and not delay_mlflow:
            self.start_mlflow()

    def start_mlflow(self):
        if not self.use_mlflow:
            return
        print('Using mlflow server at ', os.environ['MLFLOW_TRACKING_URI'], 'ref', self.ref)
        mlflow.set_experiment('hp/ec')
        meta = self.meta
        ref = self.ref
        mlflow.start_run(run_name=ref)

        meta['argv'] = ' '.join(meta['argv'])
        gitdiff = meta['gitdiff']
        gitlog = meta['gitlog']
        mlflow.log_text(gitdiff, 'gitdiff.txt')
        mlflow.log_text(gitlog, 'gitlog.txt')
        mlflow.log_text(json.dumps(meta['params'], indent=2), 'params.txt')
        del meta['gitdiff']
        del meta['gitlog']

        _to_delete = []
        for k, v in meta['params'].items():
            if len(str(v)) > 200:
                mlflow.log_text(str(v), f'params_{k}')
                _to_delete.append(k)
        for k in _to_delete:
            del meta['params'][k]
        mlflow.log_params(meta['params'])
        del meta['params']

        _to_delete = []
        for k, v in meta.items():
            if len(str(v)) > 200:
                mlflow.log_text(str(v), f'meta_{k}')
                _to_delete.append(k)
        for k in _to_delete:
            del meta[k]
        mlflow.log_params(meta)

        mlflow.set_tags(meta)
        print('stored experiment details to mlflow server')
        self.mlflow_started = True

    def log(self, logdict: Dict[str, Any], formatstr: str = None, step: int = 0):
        try:
            self.f.write(json.dumps(logdict) + '\n')
        except Exception as e:
            print('exception', e)
            for k, v in logdict.items():
                print(k, type(v))
            raise e
        self.f.flush()
        if formatstr is not None:
            print_line = formatstr.format(**logdict)
            print(print_line)
        if self.use_mlflow and self.mlflow_started:
            try:
                mlflow.log_metrics(logdict, step)
            except Exception as e:
                print(e)
                print('failed to log to mlflow')

    def close(self):
        if self.use_mlflow and self.mlflow_started:
            print('finishing mlflow run, ref', self.ref)
            try:
                mlflow.end_run()
            except Exception as e:
                print(e)
                print('failed finish mlflow run')
