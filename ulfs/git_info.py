"""
if inside a git repo, get info from that, otherwise look for gitdiff.txt and gitlog.txt files
crash if neither of these work
"""
import subprocess
import argparse
from os.path import join


def get_git_info(cmd, file, repo_dir=None):
    try:
        git_info = subprocess.check_output(cmd, stderr=subprocess.STDOUT, cwd=repo_dir).decode('utf-8')
        print('read git info using command %s' % ' '.join(cmd))
    except Exception:
        filepath = file
        if repo_dir is not None:
            filepath = join(repo_dir, filepath)
        with open(filepath, 'r') as f:
            git_info = f.read()
        print('read git info from %s' % file)
    return git_info


def get_git_log(repo_dir=None):
    return get_git_info(
        cmd=['git', 'log', '-n', '3', '--oneline', '.'],
        file='gitlog.txt',
        repo_dir=repo_dir
    )


def get_git_diff(repo_dir=None):
    return get_git_info(
        cmd=['git', 'diff', '.'],
        file='gitdiff.txt',
        repo_dir=repo_dir
    )


def get_gitinfo_dict(repo_dir=None):
    return {
        'gitdiff': get_git_diff(repo_dir=repo_dir),
        'gitlog': get_git_log(repo_dir=repo_dir)
    }


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--repo-dir', type=str)
    args = parser.parse_args()
    gitinfo_dict = get_gitinfo_dict(repo_dir=args.repo_dir)
    print(gitinfo_dict['gitdiff'][:80])
    print('')
    print(gitinfo_dict['gitlog'])
