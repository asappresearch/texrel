"""
Just collect a single image-caption pairs, bucket them, then draw from the buckets...

This script will just stream everything to a file for now (h5 of course).

And another script can then handle doing the bucketing, perhaps

All stored examples will have positive label (but we can sample examples with negation later)
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


def run(ref, seed, out_filepath, batch_size, segment_size, negation, max_n):
    random.seed(seed)
    np.random.seed(seed)

    dataset = Dataset.create(
        dtype='agreement',
        name='relational',
        worlds_per_instance=1,
        negation=negation,
        entity_counts=(4,5),
        relations=[["x-rel", "*"], ["y-rel", "*"]],
        correct_ratio=1.0,
        collision_tolerance=0.0
    )
    print('created dset')

    segment_id = 0

    dtype_str = h5py.special_dtype(vlen=str)
    out_filepath_templ = out_filepath
    out_filepath =  out_filepath_templ.format(ref=ref, i=segment_id)
    h5_out = h5py.File(expand(out_filepath), 'w')
    images_h5 = h5_out.create_dataset(
        'images', (0, 64, 64, 3), dtype=np.float32, chunks=True, maxshape=(None, 64, 64, 3))
    vocab = dataset.vocabulary('language')
    vocab_h5 = h5_out.create_dataset('vocab', (len(vocab),), dtype=dtype_str)
    for i, w in enumerate(vocab):
        vocab_h5[i] = w
    caption_shape = dataset.vector_shape(value_name='caption')[0]
    print('caption_shape', caption_shape)
    caption_wordids_h5 = h5_out.create_dataset('caption_wordids', (0, caption_shape), dtype=np.uint8, chunks=True, maxshape=(None, caption_shape))

    meta = {
        'ref': ref,
        'seed': seed
    }
    meta['segment_id'] = segment_id
    git_info_dict = {
        'gitlog': git_info.get_git_log(),
        'gitdiff': git_info.get_git_diff()
    }
    h5_utils.store_value(h5_out, 'meta', json.dumps(meta))
    h5_utils.store_value(h5_out, 'git_info', json.dumps(git_info_dict))

    b = 0
    file_start_n = 0
    start_time = time.time()
    last_print = time.time()
    segment_size_batches = segment_size // batch_size
    print('num batches in segment', segment_size_batches)
    while True:
        if b % segment_size_batches == 0 and b > 0:
            """
            create a new dump file (~2GB segments approx)
            """
            h5_out.close()
            segment_id = b // segment_size_batches
            out_filepath = out_filepath_templ.format(ref=ref, i=segment_id)
            h5_out = h5py.File(expand(out_filepath), 'w')
            meta['segment_id'] = segment_id
            h5_utils.store_value(h5_out, 'meta', json.dumps(meta))
            h5_utils.store_value(h5_out, 'git_info', json.dumps(git_info_dict))
            file_start_n = segment_id * segment_size_batches * batch_size

            images_h5 = h5_out.create_dataset(
                'images', (0, 64, 64, 3), dtype=np.float32, chunks=True, maxshape=(None, 64, 64, 3))
            vocab = dataset.vocabulary('language')
            vocab_h5 = h5_out.create_dataset('vocab', (len(vocab),), dtype=dtype_str)
            for i, w in enumerate(vocab):
                vocab_h5[i] = w
            caption_shape = dataset.vector_shape(value_name='caption')[0]
            print('caption_shape', caption_shape)
            caption_wordids_h5 = h5_out.create_dataset('caption_wordids', (0, caption_shape), dtype=np.uint8, chunks=True, maxshape=(None, caption_shape))

            print('switched to new file', out_filepath)
            start_time = time.time()
        ex_start = time.time()
        generated = dataset.generate(n=batch_size, mode='train')
        caption_ids = generated['caption']

        # print('b', b, 'batch_size', batch_size, 'file_start_n', file_start_n, 'segment_id', segment_id)
        start_N = b * batch_size - file_start_n
        end_N_excl = start_N + batch_size
        # print('start_N', start_N, 'end_N_excl', end_N_excl)

        caption_wordids_h5.resize((end_N_excl, caption_shape))
        caption_wordids_h5[start_N:end_N_excl] = caption_ids

        worlds = generated['world']
        images_h5.resize((end_N_excl, 64, 64, 3))
        images_h5[start_N:end_N_excl] = worlds

        h5_out.flush()

        if time.time() - last_print >= 3.0 or True:
            tot_time = time.time() - start_time
            eps = end_N_excl / tot_time
            spe = tot_time / end_N_excl
            print(f'done n={end_N_excl} tot_time={tot_time:.0f} eps={eps:.3f} spe={spe:.3f}')
            last_print = time.time()
        if max_n is not None and b * batch_size >= max_n:
            print('reached max_n', max_n, ' => stopping')
            break
        b += 1



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ref', type=str, required=True)
    parser.add_argument('--segment-size', type=int, default=40000)
    parser.add_argument('--max-n', type=int)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--seed', type=int, default=123)
    parser.add_argument('--negation', action='store_true')
    parser.add_argument('--out-filepath', type=str, default='~/data/shapeworld/rawstream_{ref}_{i}.h5')
    args = parser.parse_args()
    # args.out_filepath = args.out_filepath.format(ref=args.ref)
    run(**args.__dict__)
