"""
Take the results from generate2.py, and bucket by caption

initially we'll do this just for positive examples (since we only have those...), and then
we'll extend it to negative examples too

So, after learning how jda's code works, basically, the negative samples are
simply somewhat uniformly sampled from all examples, without checking whether
they really are negative examples. So, we'll do the same. (though note that
this makes the task much simpler, than if the negative examples are not
easily distinguishable by shape/color of a single object)

This version is basically heavily based on jda's code now
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

from ulfs import h5_utils
from ulfs import git_info
from ulfs.utils import expand, die


def parse_object(o):
    o_split = o.split()
    if len(o_split) == 1:
        if o == 'shape':
            return '*', '*'
        else:
            return '*', o
    else:
        if o_split[1] == 'shape':
            return o_split[0], '*'
        else:
            return o_split


def parse_caption(capt):
    """
    split into noun phrase, prep, noun phrase; thence into
    color shape prep color shape
    """
    c = capt
    assert ' not ' not in c
    try:
        if c.startswith('an '):
            c = 'a ' + c[3:]
        c = c.replace(' an ', ' a ')
        c = ' ' + c
        first_obj = c.split(' a ')[1].split(' is ')[0]
        second_obj = c.split(' a ')[2].split(' .')[0]
        s0, c0 = parse_object(first_obj)
        s1, c1 = parse_object(second_obj)
        prep = c.split(' is ')[1].split(' a ')[0].replace(' ', '-')
        parsed = [s0, c0, prep, s1, c1]
    except Exception as e:
        print('e', e)
        print('c', c)
        raise e
    return parsed


def normalize(parsed):
    if parsed[2] == 'to-the-right-of':
        parsed[2] = 'to-the-left-of'
    elif parsed[2] == 'below':
        parsed[2] = 'above'
    else:
        return parsed
    l_c = parsed[0]
    l_s = parsed[1]
    parsed[0] = parsed[3]
    parsed[1] = parsed[4]
    parsed[3] = l_c
    parsed[4] = l_s
    return parsed


# class Vectorizer(object):
#     def __init__(self):
#         self.w2i = {}
#         self.i2w = []

#     def __getitem__(self, word):
#         if word in self.w2i:
#             return self.w2i[word]
#         else:
#             self.w2i[word] = len(self.i2w)
#             self.i2w.append(word)
#             return self.w2i[word]

#     def vectorize(self, tokens):


class SkippedFileException(Exception):
    pass


class FileSegment(object):
    def __init__(self, filepath, segment_id, max_in_samples):
        self.filepath = filepath
        self.segment_id = segment_id
        try:
            self.in_h5 = h5py.File(filepath, 'r')
        except Exception as e:
            print('couldnt open file', filepath, '=>skipping')
            self.N = 0
            raise SkippedFileException()

        self.images_h5 = self.in_h5['images']
        self.meta = json.loads(h5_utils.get_value(self.in_h5, 'meta'))
        print('meta', json.dumps(self.meta, indent=2))
        self.vocab_h5 = self.in_h5['vocab']
        self.caption_wordids_h5 = self.in_h5['caption_wordids']
        print('vocab', self.vocab_h5)
        self.vocab = list(self.vocab_h5)
        print('vocab', self.vocab)
        self.N = self.images_h5.shape[0]
        print('N', self.N)
        self.idxes_by_caption = defaultdict(list)
        for n in range(self.N):
            caption_ids = self.caption_wordids_h5[n]
            caption = ' '.join([self.vocab[i] for i in list(caption_ids)])
            parsed_c = parse_caption(caption)
            parsed_c = ' '.join(normalize(parsed_c))
            self.idxes_by_caption[parsed_c].append(n)
            if max_in_samples is not None and n >= max_in_samples:
                print(f'reached max in samples {max_in_samples} => skipping remaining samples')
                break


class FileSeries(object):
    def __init__(self, ref, in_filepath_templ, max_segments, max_in_samples):
        segment_id = 0
        self.idxes_by_caption = defaultdict(list)
        self.neg_idxes_by_caption = defaultdict(list)
        file_segments = []
        while True:
            in_filepath = expand(in_filepath_templ.format(ref=ref, i=segment_id))
            if not path.isfile(in_filepath):
                if segment_id == 0:
                    in_filepath = in_filepath.replace('_0.h5', '.h5')
                    assert path.isfile(in_filepath)
                else:
                    print('all files read')
                    break
            print(in_filepath)
            try:
                file_segment = FileSegment(
                    filepath=in_filepath,
                    segment_id=segment_id,
                    max_in_samples=max_in_samples
                )
            except SkippedFileException as e:
                file_segments.append(None)
                segment_id += 1
                continue
            file_segments.append(file_segment)
            print('merging idxes by caption...')
            for k, l in file_segment.idxes_by_caption.items():
                self.idxes_by_caption[k] += [(segment_id, idx) for idx in l]
            print('...merged')
            if max_in_samples is not None:
                max_in_samples -= file_segment.N
                if max_in_samples < 0:
                    print('reached max_in_samples => breaking')
                    break
            segment_id += 1
            if max_segments is not None and segment_id >= max_segments:
                print(f'reached max segments {max_segments} => skipping remaining segments')
                break


    # parser.add_argument('--caption-set-sizes', type=str, default='train=2000,val=500,test=500')
    # parser.add_argument('--splits', type=str, default='train=train:9000,val=val:500,test=test:500,val_same=train:500,val_test=train:500')


def run(
        ref, pos_ref, seed, in_filepath, max_in_samples, max_segments, caption_set_sizes, splits,
        num_sender_pos, num_sender_neg, out_examples_json
    ):
    random.seed(seed)
    np.random.seed(seed)
    r = np.random.RandomState(seed)

    caption_set_sizes_str = caption_set_sizes
    caption_set_sizes = {}
    for split_size in caption_set_sizes_str.split(','):
        split, size = split_size.split('=')
        caption_set_sizes[split] = int(size)
    print('caption_set_sizes', caption_set_sizes)

    splits_str = splits
    splits = {}
    for split_caption_size in splits_str.split(','):
        split, caption_size = split_caption_size.split('=')
        caption_set, size = caption_size.split(':')
        splits[split] = {'caption_set': caption_set, 'size': int(size)}
    print('splits', splits)

    total_captions = np.sum([size for size in caption_set_sizes.values()]).item()
    print('total_captions', total_captions)
    die()

    in_filepath_templ = in_filepath
    segment_id = 0
    file_series = FileSeries(
        ref=pos_ref,
        in_filepath_templ=in_filepath_templ,
        max_segments=max_segments,
        max_in_samples=max_in_samples
    )

    idxes_by_caption = file_series.idxes_by_caption
    captions = list(sorted(idxes_by_caption.keys()))
    print('checking for both ok...')
    ok = []
    for c, l in file_series.idxes_by_caption.items():
        if len(l) >= num_sender_pos + 1:
            ok.append(c)
    print('ok len', len(ok))
    # ok = list(ok)

    # we'll just store the indexes for now,
    # and then fetch the images later
    meta = {
        'ref': ref,
        'pos_ref': pos_ref,
        'num_sender_pos': num_sender_pos,
        'num_sender_neg': num_sender_neg,
        'seed': seed,
        'pos_filepath_templ': in_filepath_templ,
        'num_segments': max_segments,
        'max_in_samples': max_in_samples,
        'gitlog': git_info.get_git_log(),
        'gitdiff': git_info.get_git_diff()
    }

    def draw_negative_sample(c):
        """
        uniformly sampled from all examples

        assumes that all captions in idxes_by_caption have at least one example in
        """
        caption = c
        while True:
            assert len(captions) >= 2  # in case we ran out...
            capt_idx = r.randint(len(captions))
            caption = captions[capt_idx]
            if caption == c:
                continue
            if len(idxes_by_caption[caption]) == 0:
                del captions[capt_idx]
                del idxes_by_caption[caption]
                continue
            # print('caption', caption)
            res = idxes_by_caption[caption].pop()
            if len(idxes_by_caption[caption]) == 0:
                del idxes_by_caption[caption]
                del captions[capt_idx]
            return res

    last_print = time.time()
    with open(expand(out_examples_json.format(ref=ref)), 'w') as f:
        f.write(json.dumps(meta) + '\n')
        for n in range(num_out_samples):
            c is None
            while True:
            # while c is None or len(idxes_by_caption[c]) < num_sender_pos + 1:
                ok_idx = r.randint(len(ok))
                c = ok[ok_idx]
                if len(idxes_by_caption[c]) < num_sender_pos + 1:
                    del ok[ok_idx]
                    continue
                break
                # print('len(idxes_by_caption[c])', len(idxes_by_caption[c]))

            sender_pos_exs = []
            for j in range(num_sender_pos):
                sender_pos_exs.append(idxes_by_caption[c].pop())

            sender_neg_exs = []
            for j in range(num_sender_neg):
                sender_neg_exs.append(draw_negative_sample(c))

            is_neg = r.randint(2)
            if is_neg == 1:
                receiver_ex = draw_negative_sample(c)
            else:
                receiver_ex = idxes_by_caption[c].pop()

            ex = {
                'sender_pos': sender_pos_exs,
                'sender_neg': sender_neg_exs,
                'receiver_ex': receiver_ex,
                'receiver_label': 1 - is_neg,
                'c': c
            }

            if len(idxes_by_caption[c]) < num_sender_pos + 1:
                del ok[ok_idx]

            if len(idxes_by_caption[c]) == 0:
                del idxes_by_caption[c]
                captions.remove(c)

            f.write(json.dumps(ex) + '\n')
            if n < 10:
                print(ex)
            if time.time() - last_print >= 3.0:
                print('n', n)
                last_print = time.time()
    print(f'written {num_out_samples} samples to json file {out_examples_json}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ref', type=str, required=True)
    parser.add_argument('--pos-ref', type=str, required=True)
    parser.add_argument('--max-in-samples', type=int)
    parser.add_argument('--max-segments', type=int)
    parser.add_argument('--caption-set-sizes', type=str, default='train=2000,val=500,test=500')
    parser.add_argument('--splits', type=str, default='train=train:9000,val=val:500,test=test:500,val_same=train:500,val_test=train:500')
    parser.add_argument('--num-sender-pos', type=int, default=6)
    parser.add_argument('--num-sender-neg', type=int, default=6)
    parser.add_argument('--seed', type=int, default=123)
    parser.add_argument('--in-filepath', type=str, default='~/data/shapeworld/rawstream_{ref}_{i}.h5')
    parser.add_argument('--out-examples-json', type=str, default='~/data/shapeworld/examples_{ref}_{split}.txt')
    args = parser.parse_args()
    run(**args.__dict__)
