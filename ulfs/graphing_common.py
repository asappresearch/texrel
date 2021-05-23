import subprocess
import json
from typing import Iterable, Tuple, List, Optional


def run(cmd_list, tail_lines=0):
    return '\n'.join(subprocess.check_output(cmd_list).decode('utf-8').split('\n')[- tail_lines:]).strip()


def get_recent_logfiles(path, age_minutes):
    files = subprocess.check_output(['find', path, '-cmin', '-%s' % age_minutes]).decode('utf-8').split('\n')
    files = [f for f in files if f != '' and not f.endswith('logs')]
    return files


def get_logfiles_by_pattern(path, pattern):
    cmd_list = ['ls', path]
    print(cmd_list)
    files = subprocess.check_output(cmd_list).decode('utf-8').split('\n')
    files = [f for f in files if f != '' and not f.endswith('logs') and pattern in f]
    return files


def read_meta(filepath):
    import json
    head_line = head_line = head(filepath, 1)
    meta = json.loads(head_line)
    return meta


def head(file, lines):
    return subprocess.check_output(['head', '-n', str(lines), file]).decode('utf-8')


def tail(file, lines):
    return subprocess.check_output(['tail', '-n', str(lines), file]).decode('utf-8')


def get_filepath_of_longest_file(filepaths: Iterable[str]) -> Tuple[str, int]:
    longest_filepath = ''
    max_lines = -1
    for filepath in filepaths:
        print(filepath)
        num_lines = get_num_lines(filepath)
        if num_lines > max_lines:
            max_lines = num_lines
            longest_filepath = filepath
    filepath = longest_filepath
    return filepath, max_lines


def get_ref(filepath):
    try:
        with open(filepath, 'r') as f:
            meta_line = f.readline()
        print('meta_line [' + meta_line + ']')
        if meta_line == '':
            return None
#             print('meta_line', meta_line)
        meta = json.loads(meta_line.replace('meta: ', '').strip())
#             print(meta.get('params', {}).keys())
        ref = meta.get('params', {}).get('ref', '')
        return ref
    except Exception as e:
        print('graphing_commmon.get_ref exception', e, filepath)
        return ''


def get_meta_keys(filepath: str, keys: List[str]) -> List[Optional[str]]:
    try:
        with open(filepath, 'r') as f:
            meta_line = f.readline()
        if meta_line == '':
            return [None] * len(keys)
        meta = json.loads(meta_line.replace('meta: ', '').strip())
#         params = meta.get('params', {})
        values = []
        for key in keys:
            d = meta
            key_parts = key.split('.')
            for k in key_parts[:-1]:
                # print('d.keys()', d.keys(), 'k', k)
                d = d.get(k, {})
            v = d.get(key_parts[-1], '')
            values.append(v)
        return values
    except Exception as e:
        print('graphing_common.get_meta_keys exception', e, filepath)
        return [None] * len(keys)


def get_num_lines(filepath):
    num_lines = int(subprocess.check_output(['wc', '-l', filepath]).decode('utf-8').split(' ')[0])
    return num_lines
