"""
generate shapeworld data

we will generate the following dataset:

pos=5 neg=0
pos=3 neg=2
pos=64 neg=64
pos=128 neg=0

The ones with only positive exampels will be fed to L3.
The ones with both positive and negative examples will be fed to our meta-learning algo.

for each example, we'll also have one receiver example (might be positive or negative)

This code assumes use of branch vector-correct-ratio of ShapeWorld repo, ie
https://github.com/ASAPPinc/ShapeWorld/tree/vector-correct-ratio , which allows
using lists for the correct_ratio
"""
import argparse
import os
from os import path
from os.path import join
import random
import json
import time

import numpy as np
import h5py

import torch

from shapeworld.dataset import Dataset

from ulfs import h5_utils
from ulfs import git_info
from ulfs.utils import expand, die


def run(ref, out_filepath, seed, num_pos, num_neg, num_train):
    # ratio = num_pos / (num_pos + num_neg)
    # we will generate one extra positve, and one extra negative, and choose one of these randomly
    # as receiver examples
    # ratio = (num_pos + 1) / (num_pos + 1 + num_neg + 1)
    num_send = num_pos + num_neg
    random.seed(seed)
    np.random.seed(seed)

    dtype_str = h5py.special_dtype(vlen=str)
    h5_out = h5py.File(expand(out_filepath), 'w')
    sender_images_h5 = h5_out.create_dataset(
        'sender_images', (0, num_send, 64, 64, 3), dtype=np.float32, chunks=True, maxshape=(None, num_send, 64, 64, 3))
    sender_labels_h5 = h5_out.create_dataset(
        'sender_labels', (0, num_send, ), dtype=np.uint8, chunks=True, maxshape=(None, num_send, ))
    receiver_images_h5 = h5_out.create_dataset(
        'receiver_images', (0, 64, 64, 3), dtype=np.float32, chunks=True, maxshape=(None, 64, 64, 3))
    receiver_labels_h5 = h5_out.create_dataset(
        'receiver_labels', (0, ), dtype=np.uint8, chunks=True, maxshape=(None, ))

    ratio = [1] * num_pos + [0] * num_neg + [0.5]
    print('ratio', ratio)

    dataset = Dataset.create(
        dtype='agreement',
        name='relational',
        worlds_per_instance=len(ratio),
        negation=False,
        entity_counts=(4,5),
        relations=[["x-rel", "*"], ["y-rel", "*"]],
        correct_ratio=ratio,
        collision_tolerance=0.0
    )
    print('created dset')
    print('num_send', num_send, 'ratio', ratio)
    vocab = dataset.vocabulary('language')
    vocab_h5 = h5_out.create_dataset('vocab', (len(vocab),), dtype=dtype_str)
    for i, w in enumerate(vocab):
        vocab_h5[i] = w
    caption_shape = dataset.vector_shape(value_name='caption')[0]
    print('caption_shape', caption_shape)
    caption_wordids_h5 = h5_out.create_dataset('caption_wordids', (0, caption_shape), dtype=np.uint8, chunks=True, maxshape=(None, caption_shape))

    meta = {
        'ref': ref,
        'num_pos': num_pos,
        'num_neg': num_neg,
        'num_train': num_train,
        'seed': seed
    }
    git_info_dict = {
        'gitlog': git_info.get_git_log(),
        'gitdiff': git_info.get_git_diff()
    }
    h5_utils.store_value(h5_out, 'meta', json.dumps(meta))
    h5_utils.store_value(h5_out, 'git_info', json.dumps(git_info_dict))

    n = 0
    last_print = time.time()
    while n < num_train:
        ex_start = time.time()
        generated = dataset.generate(n=1, mode='train', alternatives=True)
        caption_ids = generated['caption'][0]
        caption = dataset.to_surface(value_type='language', word_ids=caption_ids)
        print('caption', caption)
        labels = torch.ByteTensor([int(v) for v in generated['agreement'][0]])
        print('labels', labels)
        caption_wordids_h5.resize((n + 1, caption_shape))
        caption_wordids_h5[n] = caption_ids

        worlds = generated['world'][0]
        print('len(worlds)', len(worlds))
        print('worlds[0].shape', worlds[0].shape)

        shuffled_sender_idx = list(np.random.choice(num_send, num_send, replace=False))
        sender_images_h5.resize((n + 1, num_send, 64, 64, 3))
        receiver_images_h5.resize((n + 1, 64, 64, 3))
        sender_labels_h5.resize((n + 1, num_send))
        receiver_labels_h5.resize((n + 1,))

        for j in range(num_send):
            idx = shuffled_sender_idx[j].item()
            sender_images_h5[n, j] = worlds[idx]
            sender_labels_h5[n, j] = labels[idx]

        receiver_images_h5[n] = worlds[num_send]
        receiver_labels_h5[n] = labels[num_send]

        h5_out.flush()
        print('saved example time=', '%.3f' % (time.time() - ex_start))

        if time.time() - last_print >= 3.0:
            print('n', n)
            last_print = time.time()
        n += 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ref', type=str, required=True)
    parser.add_argument('--seed', type=int, default=123)
    parser.add_argument('--num-pos', type=int, default=5)
    parser.add_argument('--num-train', type=int, default=9000)
    parser.add_argument('--num-neg', type=int, default=0)
    parser.add_argument('--out-filepath', type=str, default='~/data/shapeworld/raw_{ref}.h5')
    args = parser.parse_args()
    args.out_filepath = args.out_filepath.format(ref=args.ref)
    run(**args.__dict__)
