import sys
import tempfile
import os
from os.path import join
import subprocess
import pickle
import gzip

import pytest
import torch
import numpy as np

from texrel import create_dataset


def test_name_to_hash():
    names = ['foo', 'Foo', 'Bar']
    hash_by_name = {}
    for name in names:
        hash_v = create_dataset.name_to_hash(name)
        assert hash_v not in hash_by_name.values()
        hash_by_name[name] = hash_v

    for name in names:
        hash_v = create_dataset.name_to_hash(name)
        assert hash_v == hash_by_name[name]


@pytest.mark.parametrize(
    "hypothesis_generator",
    [
        "Colors1"
    ]
)
def test_reproducible(hypothesis_generator: str):
    """
    check that if we run create_dataset twice, we get the same output file
    """
    with tempfile.TemporaryDirectory() as temp_d:
        ret = os.system(
            f'{sys.executable} texrel/create_dataset.py'
            ' --ds-ref foo'
            f' --out-filepath {temp_d}/foo_v1.dat'
            f' --hypothesis-generator {hypothesis_generator}'
            ' --num-train 32'
            ' --num-val-same 32'
            ' --num-val-new 32'
            ' --num-test-same 32'
            ' --num-test-new 32'
            ' --num-distractors 2'
            ' --num-holdout 4'
            ' --num-colors 9'
            ' --num-shapes 9')
        assert ret == 0
        file_hash = subprocess.check_output([
            'md5sum', join(temp_d, 'foo_v1.dat')
        ]).decode('utf-8').split()[0]
        print('file_hash', file_hash)

        with gzip.open(f'{temp_d}/foo_v1.dat', 'rb') as f:
            d1 = pickle.load(f)  # type: ignore
        os.system(f'ls {temp_d}')
        print('d1')
        print('d1.keys()', d1.keys())
        print(d1['meta'])

        ret = os.system(
            f'{sys.executable} texrel/create_dataset.py'
            ' --ds-ref foo'
            f' --out-filepath {temp_d}/foo_v2.dat'
            f' --hypothesis-generator {hypothesis_generator}'
            ' --num-train 32'
            ' --num-val-same 32'
            ' --num-val-new 32'
            ' --num-test-same 32'
            ' --num-test-new 32'
            ' --num-distractors 2'
            ' --num-holdout 4'
            ' --num-colors 9'
            ' --num-shapes 9')
        assert ret == 0
        file_hash2 = subprocess.check_output([
            'md5sum', join(temp_d, 'foo_v2.dat')
        ]).decode('utf-8').split()[0]
        print('file_hash2', file_hash2)

        with gzip.open(f'{temp_d}/foo_v2.dat', 'rb') as f:
            d2 = pickle.load(f)  # type: ignore
        print('d2')
        print(d2['meta'])

        for k, v1 in d1['meta'].items():
            v2 = d2['meta'][k]
            assert v1 == v2

        print('d1[data].keys()', d1['data'].keys())
        for split_name, d1_split_data in d1['data'].items():
            d2_split_data = d2['data'][split_name]
            print(split_name, d2_split_data.keys())
            for k2, d1_values in d1_split_data.items():
                print(k2)
                d2_values = d2_split_data[k2]
                if isinstance(d1_values, torch.Tensor):
                    assert (d1_values == d2_values).all()
                elif isinstance(d1_values, np.ndarray):
                    d1_values_t = torch.from_numpy(d1_values)
                    d2_values_t = torch.from_numpy(d2_values)
                    assert (d1_values_t == d2_values_t).all()
                else:
                    assert d1_values == d2_values

        assert file_hash == file_hash2
