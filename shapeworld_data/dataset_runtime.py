"""
using the new datasets, ie pos6_neg0, pos128_neg0, etc

this targets use by hypprop_l3, not by reprod/ latlang.py (which will have its own dataset impl)

So, we for now just need two datasets:

- 'training' dataset corresponds to 'train' within the shapeworld splits
- 'holdout' dataset corresponds to 'val_new' within the shapeworld splits
"""
from os.path import join, expanduser as expand
import json
import random
import argparse

import numpy as np

import torch

from ulfs.params import Params
from ulfs import utils


class Vocab(object):
    def __init__(self, w2i, vocab):
        self.w2i = w2i
        self.vocab = vocab
        self.vocab_size = len(w2i)


class Dataset(object):
    def __init__(self, data_dir, split_name):
        self.data_dir = data_dir
        self.split_name = split_name

        split_dir = join(expand(data_dir), split_name)
        self.sender_feats = torch.load(join(split_dir, 'sender_feats.pth'))
        self.sender_labels = torch.from_numpy(np.load(join(split_dir, 'sender_labels.npy')))
        self.receiver_feats = torch.load(join(split_dir, 'receiver_feats.pth'))
        self.receiver_labels = torch.from_numpy(np.load(join(split_dir, 'receiver_labels.npy')))

        with open(join(split_dir, 'hints.json'), 'r') as f:
            self.captions = json.load(f)

        self.N = self.receiver_labels.size(0)
        self.feats_size = self.receiver_feats.size(1)
        print('loaded data for', split_name, 'N', self.N, 'feats_size', self.feats_size)

    def sample(self, batch_size):
        idxes = np.random.choice(self.N, batch_size, replace=False)
        captions = [self.captions[i.item()] for i in idxes]
        res = {
            'captions': captions,
            'sender_feats': self.sender_feats[:, idxes],
            'sender_labels': self.sender_labels[idxes].transpose(0, 1),
            'receiver_feats': self.receiver_feats[idxes],
            'receiver_labels': self.receiver_labels[idxes]
        }
        return res


class Datasets(object):
    def __init__(self, ds_ref, data_dir):
        self.ds_ref = ds_ref
        self.data_dir = data_dir.format(ds_ref=ds_ref)
        with open(join(expand(self.data_dir), 'meta.json'), 'r') as f:
            self.meta = Params(json.load(f))
        print('meta', self.meta)

        self.split_by_name = {}
        self.split_by_name['train'] = Dataset(
            data_dir=self.data_dir,
            split_name='train'
        )
        self.feats_size = self.train_set.feats_size
        self.split_by_name['val_new'] = Dataset(
            data_dir=self.data_dir,
            split_name='val'
        )

    def sample(self, batch_size: int, split_name: str):
        # ds = self.train_set if training else self.holdout_set
        ds = self.split_by_name[split_name]
        return ds.sample(batch_size=batch_size)


def run(ds_ref, data_dir, seed, training):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)

    datasets = Datasets(ds_ref=ds_ref, data_dir=data_dir)
    batch_size = 4
    batch = datasets.sample(batch_size=batch_size, training=training)
    print('batch.keys()', batch.keys())
    for k, v in batch.items():
        if isinstance(v, list):
            print(k, v[:5])
        else:
            print(k, v.size())


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ds-ref', type=str, required=True)
    parser.add_argument('--seed', type=int, default=123)
    parser.add_argument('--data-dir', type=str, default='~/data/shapeworld/{ds_ref}')
    parser.add_argument('--eval', action='store_true')
    args = parser.parse_args()
    utils.reverse_args(args, 'eval', 'training')
    run(**args.__dict__)
