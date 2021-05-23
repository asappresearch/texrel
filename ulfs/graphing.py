"""
to import from jupyter:

%matplotlib inline
import sys
if '..' not in sys.path:
    sys.path.append('..')
from ulfs import graphing
import importlib
importlib.reload(ulfs.graphing)

graphing.update_index(graphing.script2ref2info)
groups = [
    {'title': '',
         'value_keys': [
             'rho', 'p_value', 'unique'
         ],
         'boxes': ['ilmb21'],
         'script': 'ilm_rnn2018janb', 'units': '', 'use_subplots': True, 'step_key': 'episode',
         'skip_record_types': ['sup_train_res']
    },
]
graphing.plot_groups2_combos(groups)
"""
from typing import Iterable, Optional
import matplotlib.pyplot as plt
import json
import math
import numpy as np
from collections import defaultdict
import tabletext

from ulfs import graphing_common
from ulfs import graphing_indexes


log_dir = '../logs'


def get_log_results(logfile, step_key, skip_record_types, value_key, record_type=None, max_step=None):
    with open(logfile, 'r') as f:
        c = f.read()
    lines = c.split('\n')
    steps = []
    values = []
    skip_record_types = set(skip_record_types)
    value_by_epoch = {}
    time_by_epoch = {}
    value_keys = value_key.split(',')
    value_key = None
    for line in lines[1:]:
        line = line.strip()
        if line == '':
            continue
        try:
            d = json.loads(line)
        except Exception as e:
            print('exception', e)
            print('line', line)
            raise e
        if record_type is not None and d.get('record_type', None) != record_type:
            continue
        if max_step is not None and d[step_key] > max_step:
            continue
        if d.get('record_type', None) in skip_record_types:
            continue
        steps.append(d[step_key])
        if value_key is None:
            for k in value_keys:
                if k in d:
                    value_key = k
                    break
            if value_key is None:
                raise Exception(
                    'no keys from ' + str(value_keys) + ' found in keys ' + str(
                        d.keys()) + ' file ' + logfile)
        values.append(d[value_key])
        value_by_epoch[d[step_key]] = d[value_key]
        time_by_epoch[d[step_key]] = d['elapsed_time']
    return steps, values, value_by_epoch, time_by_epoch


def get_log_results_multi(
        logfile: str, step_key: str, skip_record_types: Iterable[str], value_keys: Iterable[str],
        record_type: Optional[str] = None, max_step: Optional[int] = None):
    """
    returns values for multiple keys
    """
    with open(logfile, 'r') as f:
        c = f.read()
    lines = c.split('\n')
    steps = []
    values_by_key = defaultdict(list)
    skip_record_types = set(skip_record_types)
    for line in lines[1:]:
        line = line.strip()
        if line == '':
            continue
        try:
            d = json.loads(line)
        except Exception as e:
            print('exception', e)
            print('line', line)
            raise e
        if record_type is not None and d.get('record_type', None) != record_type:
            continue
        if max_step is not None and d[step_key] > max_step:
            continue
        if d.get('record_type', None) in skip_record_types:
            continue
        steps.append(d[step_key])
        for k in value_keys:
            values_by_key[k].append(d[k])
    return steps, values_by_key


def plot_logfile2(
        logfile, step_key, skip_record_types, value_key, units='thousands',
        record_type=None, y_lims=None, title=None,
        label=None, x_lims=None, max_step=None, color=None
):
    steps, values, value_by_epoch, time_by_epoch = get_log_results(
        logfile=logfile, step_key=step_key, skip_record_types=skip_record_types, value_key=value_key,
        record_type=record_type, max_step=max_step
    )
    if units in ['', 'ones', 'units']:
        divider = 1
    elif units == 'thousands':
        divider = 1000
    elif units == 'millions':
        divider = 1000 * 1000
    else:
        raise Exception('unknown units ' + units)
#     max_step = np.max(steps)
#     print('max_step', max_step)
#     divider = 1
#     units = ''
#     if max_step >= 3000:
#         divider = 1000
#         units = '(thousands)'
#     if max_step >= 3000000:
#         divider = 1000000
#         units = '(millions)'
    steps = [s / divider for s in steps]

    if y_lims is not None:
        plt.ylim(y_lims)
    if x_lims is not None:
        plt.xlim(x_lims)
    plt.plot(steps, values, label=label, color=color)
    xlabel = (step_key + ' ' + units).strip()
    plt.xlabel(xlabel)
    plt.ylabel(value_key)
    if label is not None:
        plt.legend()
    if title is not None:
        plt.title(title)
    return value_by_epoch, time_by_epoch


def plot_logfile(**kwargs):
    plt.figure(figsize=(10, 6))
    plot_logfile2(**kwargs)
    plt.show()


def plot_sequentially():
    plt.figure(figsize=(10, 6))
    files = graphing_common.get_recent_logfiles('../logs', 1)
    file_by_box = {}
    boxes = []
    for file in files:
        box = file.split('/')[-1].split('.')[0][-4:]
        boxes.append(box)
        file_by_box[box] = file
    # files = [file.split('/')[-1].split('.')[0][-4:] for file in files]
    for box in sorted(boxes):
        y_lims = None
        file = file_by_box[box]
    #     box = file.split('/')[-1].split('.')[0][-4:]
        if box == 'logs':
            continue
    #     print(box)
    #     head_line = head(file, 1)
    #     tail_line = tail(file, 1)
    #     print('')
        plot_logfile2(
            logfile=file, step_key='episode', value_key='eval_acc', title=None, y_lims=y_lims,
            label=box
        )
        if box in ['tok2']:
            plt.show()
            plt.figure(figsize=(10, 6))
    plt.show()


def plot_sequentially_2():
    files = graphing_common.get_recent_logfiles('../logs', 5)
    meta_by_file = {}
    for file in files:
        # print('file', file)
        head_line = graphing_common.head(file, 1)
        if head_line.strip() == '':
            continue
        meta = json.loads(head_line.replace('meta:', '').strip())
        meta_by_file[file] = meta
    file_by_box = {}
    boxes = []
    for file in files:
        box = file.split('/')[-1].split('.')[0][-4:]
        if box == 'logs':
            continue
        if file not in meta_by_file:
            continue
        meta = meta_by_file[file]
        ref = meta['params'].get('ref', None)
        if ref is not None:
            box = 'ref_%s' % ref
        boxes.append(box)
        file_by_box[box] = file
    print('boxes', boxes)

    plt.figure(figsize=(10, 6))
    for box in sorted(boxes):
        y_lims = None
        file = file_by_box[box]
        plot_logfile2(
            logfile=file, step_key='episode', value_key='eval_acc', title=None, y_lims=y_lims,
            label=box
        )
    plt.show()


def plot_groups(groups):
    files = graphing_common.get_recent_logfiles('../logs', 1)
    meta_by_file = {}
    for file in files:
        # print('file', file)
        head_line = graphing_common.head(file, 1)
        if head_line.strip() == '':
            continue
        meta = json.loads(head_line.replace('meta:', '').strip())
        meta_by_file[file] = meta
    file_by_box = {}
    boxes = []
    for file in files:
        box = file.split('/')[-1].split('.')[0][-4:]
        if box == 'logs':
            continue
        if file not in meta_by_file:
            continue
        meta = meta_by_file[file]
        ref = meta['params'].get('ref', None)
        if ref is not None:
            box = 'ref_%s' % ref
        boxes.append(box)
        file_by_box[box] = file
    print('boxes', boxes)

    for group_def in groups:
        # plt.figure(figsize=(10, 6))
        plt.figure(figsize=(6, 4))
        title = group_def['title']
        group_boxes = set(group_def['boxes'])
        step_key = group_def.get('step_key', 'episode')

        for box in sorted(boxes):
            if box not in group_boxes:
                continue
            y_lims = None
            file = file_by_box[box]
            plot_logfile2(
                logfile=file, step_key=step_key, value_key='eval_acc', title=None, y_lims=y_lims,
                label=box
            )
        plt.title(title)
        plt.show()


def plot_group(group, script2ref2info):
    group_def = group
    title = group_def['title']
    group_boxes = group_def['boxes']
    script = group_def['script']
    for value_key in group['value_keys']:
        plt.figure(figsize=(10, 6))
        for box in group_boxes:
            # if script2ref2info.get(script, {}).get(box, None) is None:
            #     continue
            # info = script2ref2info[script][box]
            info = graphing_indexes.get_script_info(log_dir=log_dir, box=box, script=script)
            y_lims = group_def.get('y_lims', None)
            filepath = info['filepath']
            step_key = group_def.get('step_key', 'episode')
            plot_logfile2(
                logfile=filepath, step_key=step_key, value_key=value_key, title=None, y_lims=y_lims,
                label=box, units=group['units']
            )
        plt.title(title)
        plt.show()


def plot_group_with_subplots(group):
    group_def = group
    title = group_def['title']
    group_boxes = group_def['boxes']
    script = group_def.get('script', None)
    scripts = group_def.get('scripts', None)
    value_keys = group['value_keys']
    dev_measure = group.get('dev_measure', None)
    dev_measure_max_step = group.get('dev_measure_max_step', None)
    record_type = group.get('record_type', None)
    y_lims = group_def.get('y_lims', None)
    x_lims = group_def.get('x_lims', None)
    step_key = group_def.get('step_key', 'episode')
    skip_record_types = group_def.get('skip_record_types', None)

    num_subplots = len(value_keys)
    num_cols = group_def.get('num_cols', 4)
    num_rows = (num_subplots + num_cols - 1) // num_cols
    print(title)
    value_by_epoch_by_box_by_value_key = defaultdict(dict)
    time_by_epoch_by_box_by_value_key = defaultdict(dict)
    plt.figure(figsize=(20, 14 / num_cols * num_rows))
    plt.cla()

    for i, value_key in enumerate(value_keys):
        if value_key is None:
            continue
        plt.subplot(num_rows, num_cols, i + 1)
        for box in group_boxes:
            info = graphing_indexes.get_script_info(log_dir=log_dir, box=box, script=script, scripts=scripts)
            filepath = info['filepath']
            value_by_epoch, time_by_epoch = plot_logfile2(
                logfile=filepath, step_key=step_key, value_key=value_key, title=value_key, y_lims=y_lims,
                x_lims=x_lims, record_type=record_type,
                label=box, units=group['units'], skip_record_types=skip_record_types
            )
            value_by_epoch_by_box_by_value_key[value_key][box] = value_by_epoch
            time_by_epoch_by_box_by_value_key[value_key][box] = time_by_epoch
    if dev_measure is not None:
        if dev_measure not in value_keys:
            for box in group_boxes:
                info = graphing_indexes.get_script_info(log_dir=log_dir, box=box, script=script, scripts=scripts)
                filepath = info['filepath']
                steps, values, value_by_epoch, time_by_epoch = get_log_results(
                    logfile=filepath, step_key=step_key, skip_record_types=skip_record_types, value_key=dev_measure,
                    record_type=record_type
                )
                value_by_epoch_by_box_by_value_key[dev_measure][box] = value_by_epoch
                time_by_epoch_by_box_by_value_key[dev_measure][box] = time_by_epoch
        best_dev_step_by_box = {}
        for box in group_boxes:
            best_step = 0
            best_value = -1e8
            for _e, v in value_by_epoch_by_box_by_value_key[dev_measure][box].items():
                if v > best_value and (dev_measure_max_step is None or _e <= dev_measure_max_step):
                    best_step = _e
                    best_value = v
            best_dev_step_by_box[box] = best_step
            print('box', box, 'best_step', best_step, 'best_value', best_value)
    stats_epochs = group.get('stats_epochs', [])
    results_by_epoch = defaultdict(dict)
    table_text_list = []
    table_text_list.append(['epoch'] + ['refs', 'count', 'hrs'] + value_keys)
    for e in stats_epochs:
        for value_key in value_keys:
            values = []
            times = []
            valid_refs = []
            for box in group_boxes:
                _values = [v for _e, v in value_by_epoch_by_box_by_value_key[value_key][box].items() if _e >= e]
                _times = [v for _e, v in time_by_epoch_by_box_by_value_key[value_key][box].items() if _e >= e]
                if len(_values) > 0:
                    v = _values[0]
                    if v != v:
                        # map nan to 0, for now
                        v = 0
                    values.append(v)
                    times.append(_times[0])
                    valid_refs.append(box)
            if len(values) > 0:
                mean = np.mean(values).item()
                std = np.std(values).item()
                stderr = std / math.sqrt(len(values))
                time = np.mean(times).item() / 3600
                results_by_epoch[e][value_key] = f'{mean:.3f}+/-{stderr:.3f}'
                results_by_epoch[e]['count'] = len(values)
                results_by_epoch[e]['hrs'] = f'{time:.1f}'
                results_by_epoch[e]['refs'] = ','.join(valid_refs)

    if dev_measure is not None:
        e = 'earlystop'
        stats_epochs.append(e)
        for value_key in value_keys:
            values = []
            times = []
            valid_refs = []
            for box in group_boxes:
                best_step = best_dev_step_by_box[box]
                _values = [v for _e, v in value_by_epoch_by_box_by_value_key[value_key][box].items() if _e >= best_step]
                _times = [v for _e, v in time_by_epoch_by_box_by_value_key[value_key][box].items() if _e >= best_step]
                if len(_values) > 0:
                    v = _values[0]
                    if v != v:
                        # map nan to 0, for now
                        v = 0
                    values.append(v)
                    times.append(_times[0])
                    valid_refs.append(box)
            if len(values) > 0:
                mean = np.mean(values).item()
                std = np.std(values).item()
                stderr = std / math.sqrt(len(values))
                time = np.mean(times).item() / 3600
                results_by_epoch[e][value_key] = f'{mean:.3f}+/-{stderr:.3f}'
                results_by_epoch[e]['count'] = len(values)
                results_by_epoch[e]['hrs'] = f'{time:.1f}'
                results_by_epoch[e]['refs'] = ','.join(valid_refs)

    for e in stats_epochs:
        if e in results_by_epoch:
            row = [e]
            row.append(results_by_epoch[e]['refs'])
            row.append(results_by_epoch[e]['count'])
            row.append(results_by_epoch[e]['hrs'])
            for value_key in value_keys:
                row.append(results_by_epoch[e][value_key])
            table_text_list.append(row)

    plt.tight_layout()
    plt.show()

    print(tabletext.to_text(table_text_list))


def plot_groups2(groups, value_key='eval_acc'):
    for group_def in groups:
        plot_group(group=group_def, value_key=value_key)


def plot_groups2_combos(groups):
    for group in groups:
        if group.get('use_subplots', False):
            plot_group_with_subplots(group)
        else:
            plot_group(group)

# update_index(script2ref2info)

# plot_groups2_combos()


def plot_group3(group):
    # group_boxes = group['boxes']
    script = group['script']
#     plt.figure(figsize=(6, 4))
    plt.figure(figsize=(10, 6))
    for series in group['series']:
        box, value_key = series.split('.')
        # if graphing_indexes.script2ref2info.get(script, {}).get(box, None) is None:
        #     continue
        # info = graphing_indexes.script2ref2info[script][box]
        info = graphing_indexes.get_script_info(log_dir=log_dir, box=box, script=script)
        y_lims = group.get('y_lims', None)
        filepath = info['filepath']
        plot_logfile2(
            logfile=filepath, step_key='episode', value_key=value_key, title=None, y_lims=y_lims,
            label=series, units=group['units']
        )
    plt.title(value_key)
    plt.show()


def plot_groups3(groups):
    for group in groups:
        plot_group3(group)


def print_metas():
    files = graphing_common.get_recent_logfiles('../logs', 6)
    file_by_ref = {}
    refs = []
    res_str_by_ref = {}
    ref_by_hostname = {}
    for file in files:
        ref = graphing_common.get_ref(file)
        refs.append(ref)
        file_by_ref[ref] = file
    for ref in sorted(refs):
        file = file_by_ref[ref]
        head_line = graphing_common.head(file, 1)
        tail_line = graphing_common.tail(file, 1)
        if head_line == tail_line:
            continue
        print(ref)
        meta = json.loads(head_line.replace('meta:', '').strip())
        for k in ['file', 'hostname']:
            print('    ', k, meta[k])
        ref_by_hostname[meta['hostname']] = ref
        for r in meta['argv'][1:]:
            if r.startswith('--'):
                print('    ', r.replace('--', '').replace('-', '_'), end='')
            elif r.startswith('-'):
                print('    ', r.replace('-', '').replace('-', '_'), end='')
            else:
                print(' ' + r)
        d = json.loads(tail_line)
        episode = None
        for k in ['episode', 'training_step', 'step', 'epoch']:
            episode = d.get(k, None)
            if episode is not None:
                break
        res_str = f'e={episode} '
        for k in ['acc', 'eval_acc1', 's_acc1', 's_acc2', 'r_acc1', 'r_acc2']:
            if k in d:
                v = d[k]
                res_str += f' {k}={v:.3f}'
        print(res_str)
        print(tail_line)
        res_str_by_ref[ref] = res_str
        print('')

    for ref, res_str in sorted(res_str_by_ref.items()):
        print(ref, res_str)
    print('')

    for hostname, ref in sorted(ref_by_hostname.items()):
        print(hostname, ref)


def get_hists():
    files = graphing_common.get_recent_logfiles('../hists', 60 * 24)
    for file in sorted(files):
        print(file)


# update_index(script2ref2info)
