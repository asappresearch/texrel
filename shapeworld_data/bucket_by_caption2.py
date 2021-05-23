"""
Take the results from generate2.py, and bucket by caption

From looking at how jda's code works, negative samples are
drawn approximately uniformly from all examples.

This latest version is basically heavily based on jda's code now, just basically using the already-saved
image-caption pairs, rather than calling into spatial_jda. No matter what I do, his code is about 5
times more concise and readable though :P

In this latest version, we treat the presaved files as a stream, rather than first sorting into buckets

This script just stores the indexes. 'ex_json_to_tensors.py' then fetches the images,
and writes the images out.
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


class FileSegmentStreamer(object):
    def __init__(self, filepath, segment_id):
        self.filepath = filepath
        self.segment_id = segment_id
        self.pos = 0
        try:
            self.in_h5 = h5py.File(filepath, 'r')
        except Exception as e:
            print('couldnt open file', filepath, '=>skipping')
            self.N = 0
        self.images_h5 = self.in_h5['images']
        self.meta = json.loads(h5_utils.get_value(self.in_h5, 'meta'))
        print('meta', json.dumps(self.meta, indent=2))
        self.vocab_h5 = self.in_h5['vocab']
        self.caption_wordids_h5 = self.in_h5['caption_wordids']
        # print('vocab', self.vocab_h5)
        self.vocab = list(self.vocab_h5)
        # print('vocab', self.vocab)
        self.N = self.images_h5.shape[0]
        print('N', self.N)

    def __iter__(self):
        for n in range(self.N):
            caption_ids = self.caption_wordids_h5[n]
            caption = ' '.join([self.vocab[i] for i in list(caption_ids)])
            yield (self.segment_id, n), caption


class FileSeriesStreamer(object):
    """
    yields: (segment_id, intra_segment_index), caption
    """
    def __init__(self, ref, in_filepath_templ, max_segments, max_in_samples):
        self.ref = ref
        self.in_filepath_templ = in_filepath_templ

    def __iter__(self):
        segment_id = 0
        while True:
            in_filepath = expand(self.in_filepath_templ.format(ref=self.ref, i=segment_id))
            if not path.isfile(in_filepath):
                if segment_id == 0:
                    in_filepath = in_filepath.replace('_0.h5', '.h5')
                    assert path.isfile(in_filepath)
                else:
                    print('all files read')
                    break
            print(in_filepath)
            for s_n, caption in FileSegmentStreamer(filepath=in_filepath, segment_id=segment_id):
                yield s_n, caption
            segment_id += 1


class CaptionsSet(object):
    """
    this just ends up wrapping a list... :P
    """
    def __init__(self, name, captions_l):
        self.name = name
        self.captions_l = captions_l

    def __iter__(self):
        return self.captions_l.__iter__()

    def __repr__(self):
        return f'CaptionsSet({self.name}, {len(self.captions_l)})'

    def __len__(self):
        return len(self.captions_l)

    def __getitem__(self, i):
        return self.captions_l[i]


class CaptionHarvester(object):
    """
    takes a dict of caption counts by name, and
    drinks on a feed to create appropriate CaptionsSet's
    """
    def __init__(self, captions_count_by_name):
        self.captions_count_by_name = captions_count_by_name
        self.total_captions = np.sum([size for size in captions_count_by_name.values()]).item()
        print('total_captions', self.total_captions)

    def drink(self, feed):
        i = 0
        self.captions = set()
        for s_n, caption in feed:
            self.captions.add(caption)
            if len(self.captions) >= self.total_captions:
                break
            if i % 10000 == 0:
                print(i, s_n, caption, 'len(self.captions)', len(self.captions))
            i += 1

        print('read', len(self.captions), 'captions')
        self.captions = list(self.captions)
        np.random.shuffle(self.captions)
        print('self.captions[:20]', self.captions[:20])

    def __iter__(self):
        """
        return CaptionsSet objects
        """
        for name, count in self.captions_count_by_name.items():
            _captions = self.captions[-count:]
            self.captions = self.captions[:len(self.captions) - count]
            yield name, CaptionsSet(name=name, captions_l=_captions)


def parse_caption_set_sizes(caption_set_sizes_str):
    captions_count_by_name = {}
    for name_size in caption_set_sizes_str.split(','):
        name, size = name_size.split('=')
        captions_count_by_name[name] = int(size)
    return captions_count_by_name


def parse_ds_splits(ds_splits):
    ds_splits_str = ds_splits
    split_def_by_name = {}
    for dsplit_csplit_size in ds_splits_str.split(','):
        ds_split, csplit_size = dsplit_csplit_size.split('=')
        caption_set, size = csplit_size.split(':')
        split_def_by_name[ds_split] = {'caption_set': caption_set, 'size': int(size)}
    return split_def_by_name


class NotEnoughData(Exception):
    pass


class Full(Exception):
    pass


class ExamplesSet(object):
    def __init__(self, name, captions_set_name, captions_l, size, num_sender_pos, num_sender_neg):
        self.name = name
        self.captions_set_name = captions_set_name
        self.captions_l = captions_l
        self.size = size
        self.examples = []
        self.idxes_by_caption = defaultdict(list)
        self.num_sender_pos = num_sender_pos
        self.num_sender_neg = num_sender_neg

    def draw_negative_sample(self, not_c):
        """
        avoid choosing samples from not_c caption
        (mostly to simplify counting...)
        """
        idxes_by_caption = self.idxes_by_caption
        at_least_one_available = False
        for c, l in idxes_by_caption.items():
            if c != not_c and len(l) >= 1:
                at_least_one_available = True
                break
        if not at_least_one_available:
            print([(c, len(l)) for c, l in idxes_by_caption.items()])
            raise NotEnoughData()
        while True:
            caption = self.captions_l[np.random.randint(len(self.captions_l))]
            if caption == not_c:
                continue
            if len(idxes_by_caption[caption]) == 0:
                continue
            s_n = idxes_by_caption[caption].pop()
            return s_n

    def drink(self, s, n, caption):
        self.idxes_by_caption[caption].append((s, n))
        if len(self.idxes_by_caption[caption]) >= self.num_sender_pos + 1:
            pos_idxes = self.idxes_by_caption[caption]
            sender_pos_exs = []
            for j in range(self.num_sender_pos):
                sender_pos_exs.append(pos_idxes.pop())

            sender_neg_exs = []
            for j in range(self.num_sender_neg):
                sender_neg_exs.append(self.draw_negative_sample(not_c=caption))

            is_neg = np.random.randint(2)
            if is_neg == 1:
                receiver_ex = self.draw_negative_sample(not_c=caption)
            else:
                receiver_ex = pos_idxes.pop()

            ex = {
                'sender_pos': sender_pos_exs,
                'sender_neg': sender_neg_exs,
                'receiver_ex': receiver_ex,
                'receiver_label': 1 - is_neg,
                'c': caption
            }
            # print('ex', ex)
            self.examples.append(ex)
            if len(self.examples) >= self.size:
                raise Full()

    def __iter__(self):
        return self.examples.__iter__()


class ExamplesHarvester(object):
    def __init__(self, captions_set_by_name, split_def_by_name, num_sender_pos, num_sender_neg):
        self.captions_set_by_name = captions_set_by_name
        self.split_def_by_name = split_def_by_name
        self.examples_set_by_name = {}
        self.num_sender_pos = num_sender_pos
        self.num_sender_neg = num_sender_neg
        for name, split_def in split_def_by_name.items():
            size = split_def['size']
            captions_set_name = split_def['caption_set']
            captions_l = self.captions_set_by_name[captions_set_name]
            self.examples_set_by_name[name] = ExamplesSet(
                name=name,
                captions_l=captions_l,
                captions_set_name=captions_set_name,
                size=size,
                num_sender_pos=num_sender_pos,
                num_sender_neg=num_sender_neg
            )

        self.pending_examples_sets_by_caption_set_name = defaultdict(list)
        for name, examples_set in self.examples_set_by_name.items():
            self.pending_examples_sets_by_caption_set_name[examples_set.captions_set_name].append(examples_set)

        self.captions_set_name_by_caption = {}
        for name, captions_set in captions_set_by_name.items():
            for caption in captions_set:
                self.captions_set_name_by_caption[caption] = name

    def drink(self, feed):
        for (s, n), caption in feed:
            captions_set_name = self.captions_set_name_by_caption.get(caption, None)
            if captions_set_name is None:
                continue

            examples_set_l = self.pending_examples_sets_by_caption_set_name[captions_set_name]
            if len(examples_set_l) == 0:
                continue

            examples_set = examples_set_l[-1]
            try:
                examples_set.drink(s, n, caption)
            except Full as e:
                examples_set_l.pop()
                print('completed', examples_set.name)
                sets_left = np.sum([len(l) for l in self.pending_examples_sets_by_caption_set_name.values()]).item()
                print('sets_left', sets_left)
                if sets_left == 0:
                    print('all examples created :)')
                    return

    def __iter__(self):
        return self.examples_set_by_name.items().__iter__()


def run(
        ref, pos_ref, seed, in_filepath, max_in_samples, max_segments, caption_set_sizes, ds_splits,
        num_sender_pos, num_sender_neg, out_examples_json
    ):
    random.seed(seed)
    np.random.seed(seed)
    r = np.random.RandomState(seed)

    meta = {
        'ref': ref,
        'caption_set_sizes': caption_set_sizes,
        'ds_splits': ds_splits,
        'pos_ref': pos_ref,
        'num_sender_pos': num_sender_pos,
        'num_sender_neg': num_sender_neg,
        'seed': seed,
        'pos_filepath_templ': in_filepath,
        'num_segments': max_segments,
        'max_in_samples': max_in_samples,
        'gitlog': git_info.get_git_log(),
        'gitdiff': git_info.get_git_diff()
    }

    image_caption_stream = FileSeriesStreamer(ref=pos_ref, in_filepath_templ=in_filepath, max_segments=max_segments, max_in_samples=max_in_samples)

    captions_count_by_name = parse_caption_set_sizes(caption_set_sizes)
    caption_harvester = CaptionHarvester(captions_count_by_name=captions_count_by_name)
    caption_harvester.drink(image_caption_stream)
    captions_set_by_name = dict(caption_harvester)
    print('captions_set_by_name', captions_set_by_name)

    split_def_by_name = parse_ds_splits(ds_splits)
    print('split_def_by_name', split_def_by_name)

    examples_harvester = ExamplesHarvester(
        captions_set_by_name=captions_set_by_name,
        split_def_by_name=split_def_by_name,
        num_sender_pos=num_sender_pos,
        num_sender_neg=num_sender_neg
    )
    examples_harvester.drink(image_caption_stream)

    last_print = time.time()
    for name, examples_set in examples_harvester:
        filepath = expand(out_examples_json.format(ref=ref, split=name))
        print('writing', filepath)
        with open(filepath, 'w') as f:
            meta['split'] = name
            f.write(json.dumps(meta) + '\n')

            for ex in examples_set:
                f.write(json.dumps(ex) + '\n')
    print('all done')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ref', type=str, required=True)
    parser.add_argument('--pos-ref', type=str, required=True)
    parser.add_argument('--max-in-samples', type=int)
    parser.add_argument('--max-segments', type=int)
    parser.add_argument('--caption-set-sizes', type=str, default='train=2000,val=500,test=500')
    parser.add_argument('--ds-splits', type=str, default='train=train:9000,val=val:500,test=test:500,val_same=train:500,test_same=train:500')
    # parser.add_argument('--caption-set-sizes', type=str, default='train=50,val=20,test=20')
    # parser.add_argument('--ds-splits', type=str, default='train=train:90,val=val:5,test=test:5,val_same=train:5,val_test=train:5')
    parser.add_argument('--num-sender-pos', type=int, default=6)
    parser.add_argument('--num-sender-neg', type=int, default=6)
    parser.add_argument('--seed', type=int, default=123)
    parser.add_argument('--in-filepath', type=str, default='~/data/shapeworld/rawstream_{ref}_{i}.h5')
    parser.add_argument('--out-examples-json', type=str, default='~/data/shapeworld/examples_{ref}_{split}.txt')
    args = parser.parse_args()
    run(**args.__dict__)
