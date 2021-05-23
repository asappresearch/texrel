from os.path import join
import glob
from collections import defaultdict

from ulfs import graphing_common


def get_script_info(log_dir, box, script=None, scripts=None):
    # files = graphing_common.run(['ls', '-rt', f'{log_dir}/*{script}*{box}*'], 0).split('\n')
    files = []
    # assert script is not None or scripts is not None
    # print('log_dir', log_dir, 'script', script, 'box', box)
    if script is not None:
        files += glob.glob(f'{log_dir}/log_{script}_{box}_*.log')
    # print('files[:10]', files[:10])
    if scripts is not None:
        for _script in scripts:
            files += glob.glob(f'{log_dir}/log_{_script}_{box}_*.log')
    if scripts is None and script is None:
        files += glob.glob(f'{log_dir}/log_*_{box}_*.log')
        scripts = ['']
    script2ref2info = defaultdict(dict)
    for file in files:
        index_file(script2ref2info, join(log_dir, file))
    if script is not None:
        if script in script2ref2info:
            return script2ref2info[script][box]
        else:
            print('missing', box, script)
            raise Exception('missing ' + box + ', ' + script)
    for script in scripts:
        if len(script2ref2info[script]) > 0:
            return script2ref2info[script][box]
    print('script', script, 'scripts', scripts, 'box', box)
    print('files', files)
    raise Exception('not found')


def index_file(script2ref2info, filepath):
    ref, script = graphing_common.get_meta_keys(filepath, ['params.ref', 'file'])
#     print('ref', ref, 'script', script)
    if ref == '' or ref is None or script == '' or script is None:
        return
    num_lines = graphing_common.get_num_lines(filepath)
    existing_info = script2ref2info.get(script, {}).get(ref, None)
    if existing_info is None or existing_info['num_lines'] < num_lines:
        script2ref2info[script][ref] = {'filepath': filepath, 'num_lines': num_lines, 'script': script, 'ref': ref}
    existing_info = script2ref2info.get('', {}).get(ref, None)
    if existing_info is None or existing_info['num_lines'] < num_lines:
        script2ref2info[''][ref] = {'filepath': filepath, 'num_lines': num_lines, 'script': script, 'ref': ref}


def build_index(log_dir='../logs'):
    files = graphing_common.run(['ls', '-rt', log_dir], 0).split('\n')
    script2ref2info = defaultdict(dict)
    for file in files:
        index_file(script2ref2info, join(log_dir, file))
#         ref = get_ref(join(log_dir, file))
#         if ref == '' or ref is None:
#             continue
#         num_lines = get_num_lines(filepath)
#         if ref not in ref2info or num_lines > ref2info[ref]['num_lines']:
#             ref2file[ref] = {'file': file, 'num_lines': get_num_lines(filepath)}
    print('done indexing')
    print('len(script2ref2info)', len(script2ref2info))
    return script2ref2info


def update_index(script2ref2info, hours=24, log_dir='../logs'):
    files = graphing_common.get_recent_logfiles(path=log_dir, age_minutes=hours * 60)
#     print('updating from', len(files), 'files')
    for file in files:
        index_file(script2ref2info, join(log_dir, file))
#         ref = get_ref(join(log_dir, file))
#         if ref != '' and ref is not None:
#     print('updated from', len(files), 'files, in ', time.time() - start_time, ' seconds')
