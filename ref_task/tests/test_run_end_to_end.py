"""
sanity check that hypprop at least runs
"""
import sys
import os
from os.path import join
import tempfile


def test_e2e():
    with tempfile.TemporaryDirectory() as temp_d:
        # res = os.system(
        #     f'{sys.executable} texrel/thingset_create.py '
        #     f'--ref tsfoo1 --out-filepath {join(temp_d, "tsfoo1.json")}')
        # assert res == 0
        res = os.system(
            f'{sys.executable} texrel/create_dataset.py '
            '--ds-ref ds1 '
            f'--out-filepath  {join(temp_d, "ds1.dat")} '
            '--num-train 16  --num-val-same 4 '
            '--num-val-new 4 --num-test-same 4 --num-test-new 4')
        assert res == 0
        res = os.system(
            f'{sys.executable} ref_task/run_end_to_end.py --disable-cuda --ref foo '
            f'--ds-filepath-templ {join(temp_d, "ds1.dat")} --render-every-seconds 1 '
            '--max-steps 4 --batch-size 2')
        assert res == 0
