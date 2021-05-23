import os
import time
from os import path
import gzip
import pickle
import subprocess

import torch


def ensure_dirs_exist(dirs):
    for d in dirs:
        if not path.isdir(d):
            os.makedirs(d)
            print('created directory [%s]' % d)


def safe_save(filepath, target):
    """
    if out of disk space, will crash instead of erasing existing file
    """
    print('saving... ', end='', flush=True)
    save_start = time.time()
    with open(filepath + '.tmp', 'wb') as f:
        torch.save(target, f)
    os.rename(filepath + '.tmp', filepath)
    print('saved in %.1f seconds' % (time.time() - save_start))


def get_date_ordered_files(target_dir):
    files = subprocess.check_output(['ls', '-rt', target_dir]).decode('utf-8').split('\n')
    files = [f for f in files if f != '']
    return files


def safe_save_pickled_gzip(filepath, target):
    """
    - safe here means: if out of disk space, will crash instead of erasing existing file
    - saves a gzipped pickle file
        - sets filename and mtime to '' and 0 respectively so md5sum is
          repeatable
        - compress = 5 gives good compromise between speed and compression
          ratio (> 7 way too slow. 0 is no compression. anywhere from 1 to 7 is ok-ish)
    """
    trial_save_path = filepath + '.tmp'
    save_start = time.time()
    with open(trial_save_path, 'wb') as f:
        with gzip.GzipFile(
                fileobj=f, filename='', mode='wb', mtime=0, compresslevel=5) as fg:
            pickle.dump(target, fg, protocol=-1)
    os.rename(filepath + '.tmp', filepath)
    elapsed_time = time.time() - save_start
    size = os.path.getsize(filepath)
    print(f'saved in {elapsed_time:.1f} seconds, size {size // 1024 // 1024}MB')
