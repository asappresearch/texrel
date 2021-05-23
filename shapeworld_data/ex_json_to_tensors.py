"""
- take the output of bucket_by_caption.py, which is a text file with one example per line,
as json serialized dicts; and also the raw outputs from shapeworld generator, from generate2.py
- fetch the images
- write the images to ... h5 ? files numpy files? torch files?
"""
import argparse
import os
from os import path
from os.path import join
import random
from collections import defaultdict
import json
import time

import numpy as np
import h5py

import torch

from shapeworld.dataset import Dataset

from ulfs import h5_utils, utils, git_info, file_utils
from ulfs.params import Params
from ulfs.utils import expand, die

from shapeworld_data import bucket_by_caption


def run(
        ex_ref, in_examples_json, max_examples, split_names, out_torch
    ):
    out_torch = out_torch.format(ref=ex_ref)

    ds_by_split = {}
    for split in split_names.split(','):
        with open(expand(in_examples_json.format(ref=ex_ref, split=split)), 'r') as f:
            meta_dict =json.loads(f.readline())
            del meta_dict['gitlog']
            del meta_dict['gitdiff']
            meta =  Params(meta_dict)
            print(meta)
            examples = [json.loads(l) for l in f]
        if max_examples is not None:
            examples = examples[:max_examples]
        print(f'read {len(examples)} examples')

        random.seed(meta.seed)
        np.random.seed(meta.seed)
        torch.manual_seed(meta.seed)

        # sort the required images into segment sets
        n_by_segment = defaultdict(set)
        for ex in examples:
            for segment_id, idx in ex['sender_pos']:
                n_by_segment[segment_id].add(idx)
            for segment_id, idx in ex['sender_neg']:
                n_by_segment[segment_id].add(idx)
            r_s, r_i = ex['receiver_ex']
            n_by_segment[r_s].add(r_i)

        print('n_by_segment:')
        for segment, s in n_by_segment.items():
            print('    ', segment, len(s))
        print('')

        n_by_segment = {segment: sorted(list(s)) for segment, s in n_by_segment.items()}

        for segment, l in n_by_segment.items():
            print(segment, l[:20])

        def fetch_images(image_idxes_by_segment, filepath_templ, ref):
            """
            input: image_idxes_by_segment

            output:
            dictionary of images keyed on (segment_id, image_idx)
            """
            images_dict = {}
            for segment, l in image_idxes_by_segment.items():
                f_h5 = h5py.File(expand(filepath_templ.format(ref=ref, i=segment)))
                images_h5 = f_h5['images']

                _time = time.time()
                images = images_h5[l]
                print('images.shape', images.shape, 'fetched in %.1f seconds' % (time.time() - _time))
                for returned_idx, image_idx in enumerate(l):
                    images_dict[(segment, image_idx)] = images[returned_idx]
            return images_dict

        pos_images_dict = fetch_images(
            image_idxes_by_segment=n_by_segment,
            filepath_templ=meta.pos_filepath_templ,
            ref=meta.pos_ref
        )
        # print('pos_images_dict.keys()', pos_images_dict.keys())

        N = len(examples)
        sender_examples = torch.zeros(
            meta.num_sender_pos + meta.num_sender_neg, N, 64, 64, 3
        )
        sender_labels = torch.zeros(
            meta.num_sender_pos + meta.num_sender_neg, N, dtype=torch.uint8
        )
        receiver_examples = torch.zeros(
            N, 64, 64, 3
        )
        receiver_labels = torch.zeros(
            N, dtype=torch.uint8
        )

        import matplotlib.pyplot as plt
        plt.switch_backend('agg')

        captions = [ex['c'] for ex in examples]
        for n, ex in enumerate(examples):
            sender_labels_l = [1] * meta.num_sender_pos + [0] * meta.num_sender_neg
            receiver_label = ex['receiver_label']
            # print('receiver_label', receiver_label)
            M = len(sender_labels_l)
            # print(ex['c'])
            shuffled_idxes = np.random.choice(M, M, replace=False)
            _sender_images = torch.zeros(M, 64, 64, 3)
            m = 0
            sender_labels_t = torch.zeros(M, dtype=torch.uint8)
            for i, label in enumerate(sender_labels_l):
                sender_labels_t[shuffled_idxes[i]] = label
            # print('sender_labels_t', sender_labels_t)
            sender_pos = ex['sender_pos']
            sender_neg = ex['sender_neg']
            receiver_s, receiver_c = ex['receiver_ex']
            receiver_sc = (receiver_s, receiver_c)
            receiver_label = ex['receiver_label']
            for i in range(meta.num_sender_pos):
                s, c = ex['sender_pos'][i]
                image = pos_images_dict[(s, c)]
                _sender_images[shuffled_idxes[m]] = torch.from_numpy(image)
                m += 1

            for i in range(meta.num_sender_neg):
                s, c = ex['sender_neg'][i]
                image = pos_images_dict[(s, c)]
                _sender_images[shuffled_idxes[m]] = torch.from_numpy(image)
                m += 1

            receiver_image = pos_images_dict[receiver_sc]
            receiver_examples[n] = torch.from_numpy(receiver_image)
            receiver_labels[n] = receiver_label

            sender_examples[:, n] = _sender_images
            # print('sender_labels.size()', sender_labels.size())
            # print('n', n, type(n))
            # print('sender_labels_t.size()', sender_labels_t.size(), sender_labels_t.dtype)
            sender_labels[:, n] = sender_labels_t

        sender_examples = sender_examples.transpose(-2, -1).transpose(-3, -2).transpose(0, 1)
        print('sender_examples.size()', sender_examples.size())
        print('receiver_examples.size()', receiver_examples.size())
        ds_by_split[split] = {
            'sender_images': sender_examples,
            'receiver_images': receiver_examples,
            'sender_labels': sender_labels,
            'receiver_labels': receiver_labels,
            'captions': captions
        }
    ds = {
        'meta': meta.__dict__,
        'ds_by_split': ds_by_split
        # 'sender': sender_examples,
        # 'receiver': receiver_examples
    }
    file_utils.safe_save(expand(out_torch), ds)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ex-ref', type=str, required=True)
    parser.add_argument('--split-names', type=str, default='train,test,val,val_same,test_same')
    parser.add_argument('--in-examples-json', type=str, default='~/data/shapeworld/examples_{ref}_{split}.txt')
    parser.add_argument('--max-examples', type=int, help='mostly for dev purposes')
    parser.add_argument('--out-torch', type=str, default='~/data/shapeworld/ds_{ref}.pth')
    args = parser.parse_args()
    run(**args.__dict__)
