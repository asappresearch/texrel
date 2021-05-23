"""
designed to be imported from jupyter like:

%matplotlib inline

import sys, importlib
sys.path.append('..')

import ulfs
from ulfs import graphing, graphing_common, graphing_indexes, random_walk_graphs
importlib.reload(ulfs.graphing)
importlib.reload(ulfs.graphing_common)
importlib.reload(ulfs.graphing_indexes)
importlib.reload(ulfs.random_walk_graphs)

log_dir = '../logs'

refs = ['ilmc141']
for ref in refs:
    random_walk_graphs.diag_plots([ref])

"""

import matplotlib.pyplot as plt
from typing import List
import glob
import json
from os.path import join
import scipy.stats
from ulfs import graphing_common
from ulfs.params import Params


log_dir = '../logs'


def run_graphs(scripts, refs, prev_measure, step_key, next_measure, add_linear=False):
    prev_v_l = []
    next_v_l = []

    comms_dropout = None
    dropout = None

    for ref in refs:
        e2e_logfiles = []
        for script in scripts:
            e2e_logfiles += glob.glob(f'{log_dir}/log_{script}_{ref}_*.log')
        e2e_logfiles.sort()

        for i, file in enumerate(e2e_logfiles):
            log_filepath = join(log_dir, file)
            num_lines = graphing_common.get_num_lines(log_filepath)
            if num_lines < 2:
                continue
            meta = graphing_common.read_meta(log_filepath)
            meta['num_lines'] = num_lines
            params = Params(meta['params'])
            if params.ref != ref:
                continue
            meta['log_filepath'] = log_filepath
            if comms_dropout is None:
                comms_dropout = params.__dict__.get('comms_dropout', 0)
                dropout = params.dropout
            else:
                assert dropout == params.dropout
                assert comms_dropout == params.__dict__.get('comms_dropout', 0)
            prev_v = None
            with open(log_filepath, 'r') as f_in:
                for n, line in enumerate(f_in):
                    if n == 0:
                        continue
                    d = json.loads(line)
                    if d.get('record_type', 'ilm') != 'ilm':
                        continue
                    next_v = d[next_measure]
                    next_v = next_v if next_v == next_v else 0
                    new_prev_v = d[prev_measure]
                    new_prev_v = new_prev_v if new_prev_v == new_prev_v else 0
                    if prev_v is not None:
                        prev_v_l.append(prev_v)
                        next_v_l.append(next_v)
                    prev_v = new_prev_v

    plt.xlabel('previous generation ' + prev_measure)
    plt.ylabel('this generation ' + next_measure)
    plt.title(','.join(refs))
    plt.scatter(prev_v_l, next_v_l)
    if prev_measure == next_measure and add_linear:
        plt.plot(prev_v_l, prev_v_l)

    rho, p = scipy.stats.spearmanr(a=prev_v_l, b=next_v_l)
    return rho


value_keys = ['e2e_acc', 'rho', 'holdout_acc']
target_episode = 0

filters: List[str] = []
max_step = 25000
units = ''
num_rows = 2


def diag_plots(scripts, refs, measures=['e2e_acc', 'holdout_acc', 'rho'], step_key='episode'):
    scores = []
    plt.figure(figsize=(20, 3.5))
    for i, measure in enumerate(measures):
        plt.subplot(1, 4, i + 1)
        rho = run_graphs(scripts=scripts, refs=refs,
                         prev_measure=measure, next_measure=measure, add_linear=True, step_key=step_key)
        scores.append((measure, measure, rho))
    plt.show()
    for score in scores:
        print(score[0], score[1], '%.2f' % score[2])


def all_plots(scripts, refs, step_key='episode'):
    scores = []
    print('-------------------------------')
    for prev_measure in ['e2e_acc', 'holdout_acc', 'rho']:
        plt.figure(figsize=(20, 3.5))
        for i, next_measure in enumerate(['e2e_acc', 'holdout_acc', 'rho']):
            plt.subplot(1, 4, i + 1)
            rho = run_graphs(scripts=scripts, refs=refs,
                             prev_measure=prev_measure, next_measure=next_measure, add_linear=False, step_key=step_key)
            scores.append((prev_measure, next_measure, rho))
        plt.show()
    print('-------------------------------')
    print('')
    for score in scores:
        print(score[0], score[1], '%.2f' % score[2])
