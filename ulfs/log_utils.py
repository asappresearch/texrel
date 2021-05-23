from collections import defaultdict
import subprocess
import json
import argparse
from os.path import join


def run(cmd_list, tail_lines=0):
    return '\n'.join(subprocess.check_output(cmd_list).decode('utf-8').split('\n')[- tail_lines:]).strip()


def get_meta_keys(filepath, keys):
    try:
        with open(filepath, 'r') as f:
            meta_line = f.readline()
        meta = json.loads(meta_line.replace('meta: ', '').strip())
        values = []
        for key in keys:
            d = meta
            key_parts = key.split('.')
            for k in key_parts[:-1]:
                d = d.get(k, {})
            v = d.get(key_parts[-1], '')
            values.append(v)
        return values
    except Exception as e:
        print('exception', e, filepath)
        return [None] * len(keys)


def get_num_lines(filepath):
    # print(filepath)
    num_lines = int(subprocess.check_output(['wc', '-l', filepath]).decode('utf-8').strip().split(' ')[0])
    return num_lines


def index_file(script2ref2info, filepath):
    ref, script = get_meta_keys(filepath, ['params.ref', 'file'])
#     print('ref', ref, 'script', script)
    if ref == '' or ref is None or script == '' or script is None:
        return
    num_lines = get_num_lines(filepath)
    existing_info = script2ref2info.get(script, {}).get(ref, None)
    if existing_info is None or existing_info['num_lines'] < num_lines:
        script2ref2info[script][ref] = {'filepath': filepath, 'num_lines': num_lines, 'script': script, 'ref': ref}


def build_index(log_dir):
    files = run(['ls', '-rt', log_dir], 0).split('\n')
    print('building index...')
    script2ref2info = defaultdict(dict)
    for file in files:
        index_file(script2ref2info, join(log_dir, file))
    print('done indexing')
    print('len(script2ref2info)', len(script2ref2info))
    return script2ref2info


def find_ref(log_dir, file, ref):
    script2ref2info = build_index(log_dir)
    log_info = script2ref2info[file][ref]
    # print(log_info)
    return log_info


def main(log_dir, file, ref):
    log_info = find_ref(log_dir=log_dir, ref=ref, file=file)
    print(log_info)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--log_dir', type=str, default='logs')
    parser.add_argument('--file', type=str, required=True)
    parser.add_argument('--ref', type=str, required=True)
    args = parser.parse_args()
    main(**args.__dict__)
