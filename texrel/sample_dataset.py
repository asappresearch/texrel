#!/usr/bin/env python3
"""
draw samples from a dataset, and save them as images, so we can check the dataset isnt
completely broken :P
"""
import argparse
import os
from os import path
from typing import List, Optional
import json

import torch

from ulfs import image_utils
from texrel.dataset_runtime import TexRelDataset


def run(ds_refs: List[str], ds_filepath_templ: str, ds_seed: int,
        ds_texture_size: int, ds_background_noise: float, ds_mean: float, ds_mean_std: float,
        up_sample: int, out_dir: str, out_ground: str,
        num_examples: int, out_tiled_images: Optional[str], out_labels: Optional[str],
        out_hypothesis_shapes: Optional[str], out_prepositions: Optional[str],
        ds_val_refs: Optional[List[str]]):
    ds = TexRelDataset(
        ds_refs=ds_refs,
        ds_filepath_templ=ds_filepath_templ,
        ds_seed=ds_seed,
        ds_texture_size=ds_texture_size,
        ds_background_noise=ds_background_noise,
        ds_mean=ds_mean,
        ds_mean_std=ds_mean_std,
        ds_val_refs=ds_val_refs,
    )
    if not path.exists(out_dir):
        os.makedirs(out_dir)
    N = 32
    print('meta', json.dumps(ds.meta.__dict__, indent=2))
    for split_name in ['train', 'val_same', 'val_new', 'test_same', 'test_new']:
        print('')
        print('==================================================')
        print('===  ' + split_name + '  ===================================')
        print('==================================================')
        # split_name = 'val_new' if sample_holdout else 'train'
        b = ds.sample(batch_size=N, split_name=split_name, add_ground=True)
        print('b.keys()', b.keys())
        train_labels = b['inner_train_labels_t']
        test_labels = b['inner_test_labels_t']
        M_train = b['inner_train_examples_t'].size(0)
        print('M_train', M_train)
        M_test = b['inner_test_examples_t'].size(0)
        print('M_test', M_test)
        grid_planes = ds.meta.grid_planes
        grid_size = ds.meta.grid_size
        print('grid_planes', grid_planes, 'grid_size', grid_size)

        ground = b['ground_examples_t'][0]
        print('ground.size()', ground.size())
        ground = ground.transpose(0, 1).contiguous().view(ground.size(1), -1, ground.size(-1))
        print('ground.size()', ground.size())
        image_utils.save_image(filepath=out_ground.format(split_name=split_name), image=ground)

        # tiled image will have examples as rows, train examples as first columns,
        # and test examples as remaining columns
        tiled_image = torch.full((
            grid_planes,
            num_examples * (grid_size + 2),
            (M_train + M_test) * (grid_size + 2)), 0.4)
        # hypothesis_shapes = torch.full((
        #     grid_planes,
        #     num_examples * (ds_texture_size + 1) + 1,
        #     3 * (ds_texture_size + 1) + 1), 0.4)
        if not path.exists('html'):
            os.makedirs('html')
        prepositions_l: List[str] = []
        print('b.keys()', b.keys())
        for n in range(num_examples):
            print('example', n)
            print('    eng', b['hypotheses_english'][n])
            print('    dep', b['hypotheses_structured'][n])
            print('    labels',
                  'inner_train',
                  b['inner_train_labels_t'][:, n].tolist(),
                  'inner_test',
                  b['inner_test_labels_t'][:, n].tolist())
            print('hypothesis', b['hypotheses_english'][n])
            # hypothesis = b['hypotheses_t'][:, n]
            # s1, c1, prep, s2, c2 = hypothesis.tolist()
            # c1 -= 9
            # c2 -= 9
            # prep = prep - 9 * 2 - 1
            # texture_idxes = torch.tensor([s1, s2]).unsqueeze(0)

            # for m in range(3):
            #     color_idxes = torch.tensor([c1, c2]).unsqueeze(0)
            #     rel_images = ds.texturizer.forward(texture_idxes, color_idxes)
            #     hypothesis_shapes[
            #         :,
            #         n * (ds_texture_size + 1) + 1: (n + 1) * (ds_texture_size + 1),
            #         m * (ds_texture_size + 1) + 1: (m + 1) * (ds_texture_size + 1)
            #     ] = rel_images[:, :, m * ds_texture_size:(m+1) * ds_texture_size]
            # preposition = texrel.relations.Preposition.eat_from_indices(prep_space=prep_space, indices=[prep])[0]
            # print('    ' + str(preposition))
            # prepositions_l.append(str(preposition))
            print('labels', train_labels[:, n].tolist(), test_labels[:, n].tolist())
            for m in range(M_train):
                label = train_labels[m, n].item()
                color_t = torch.tensor([0, 1, 0]) if label == 1 else torch.tensor([1, 0, 0])
                tiled_image[
                    :,
                    n * (grid_size + 2) + 0: (n + 1) * (grid_size + 2) + 0,
                    m * (grid_size + 2) + 0: (m + 1) * (grid_size + 2) + 0
                ] = color_t.unsqueeze(-1).unsqueeze(-1)
                tiled_image[
                    :,
                    n * (grid_size + 2) + 1: (n + 1) * (grid_size + 2) - 1,
                    m * (grid_size + 2) + 1: (m + 1) * (grid_size + 2) - 1
                ] = b['inner_train_examples_t'][m, n]
            for m in range(M_test):
                label = test_labels[m, n].item()
                color_t = torch.tensor([0, 1, 0]) if label == 1 else torch.tensor([1, 0, 0])
                tiled_image[
                    :,
                    n * (grid_size + 2) + 0: (n + 1) * (grid_size + 2) + 0,
                    (M_train + m) * (grid_size + 2) + 0: (M_train + m + 1) * (grid_size + 2) + 0
                ] = color_t.unsqueeze(-1).unsqueeze(-1)
                tiled_image[
                    :,
                    n * (grid_size + 2) + 1: (n + 1) * (grid_size + 2) - 1,
                    (M_train + m) * (grid_size + 2) + 1: (M_train + m + 1) * (grid_size + 2) - 1
                ] = b['inner_test_examples_t'][m, n]
        print(' '.join(prepositions_l))
        print('ds_refs[:32]', b['dsrefs_t'][:32].tolist())
        if up_sample != 1:
            tiled_image = image_utils.upsample_image(up_sample=up_sample, tgt=tiled_image)
        if out_tiled_images is not None:
            image_utils.save_image(out_tiled_images.format(split_name=split_name), tiled_image)
        if out_prepositions is not None:
            with open(out_prepositions.format(split_name=split_name), 'w') as f:
                f.write('{' + ', '.join([f'"{prep}"' for prep in prepositions_l]) + '}\n')
        # if out_hypothesis_shapes is not None:
        #     hypothesis_shapes = image_utils.upsample_image(up_sample=up_sample, tgt=hypothesis_shapes)
        #     image_utils.save_image(out_hypothesis_shapes.format(split_name=split_name), hypothesis_shapes)
        labels_rows = []
        for n in range(num_examples):
            labels_row_str = '{'
            labels_both = train_labels[:, n].tolist() + test_labels[:n].tolist()
            labels_str_l = ['true' if label == 0 else 'false' for label in labels_both]
            labels_row_str += ', '.join([f'"{label}"' for label in labels_str_l])
            labels_row_str += '}'
            labels_rows.append(labels_row_str)
        out_labels_str = '{' + ', '.join(labels_rows) + '}'
        print('out_labels_str', out_labels_str)
        if out_labels is not None:
            with open(out_labels.format(split_name=split_name), 'w') as f:
                f.write(out_labels_str + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ds-texture-size', type=int, default=4)
    parser.add_argument('--up-sample', type=int, default=1, help='up sample each pixel to this many pixels')
    parser.add_argument('--ds-seed', type=int, default=123)
    parser.add_argument('--ds-refs', type=str, required=True)
    parser.add_argument('--ds-val-refs', type=str)
    parser.add_argument('--ds-filepath-templ', type=str, default='~/data/reftask/{ds_ref}.dat')
    parser.add_argument('--ds-background-noise', type=float, default=0, help='std of noise (with mean 0.5)')
    parser.add_argument('--ds-mean', type=float, default=0)
    parser.add_argument('--ds-mean-std', type=float, default=0)
    parser.add_argument('--num-examples', type=int, default=8)

    parser.add_argument('--out-dir', type=str, default='html/samples')
    parser.add_argument('--out-tiled-images', default='{out_dir}/tiled_{split_name}.png', type=str,
                        help='tile images across single image')
    parser.add_argument('--out-hypothesis-shapes', default='{out_dir}/shapes_{split_name}.png', type=str,
                        help='the shapes from the hypotheses')
    parser.add_argument('--out-ground', type=str, default='{out_dir}/ground_{split_name}.png',
                        help='ground truth images, i.e. without distractors or negatives')
    parser.add_argument('--out-prepositions', type=str, default='{out_dir}/preps_{split_name}.txt',
                        help='the prepositions from the hypotheses')
    parser.add_argument('--out-labels', type=str, default='{out_dir}/labels_{split_name}.txt')
    args = parser.parse_args()
    args.ds_refs = args.ds_refs.split(',')
    if args.ds_val_refs is not None:
        args.ds_val_refs = args.ds_val_refs.split(',')
    for k in ['out_tiled_images', 'out_hypothesis_shapes', 'out_prepositions', 'out_labels', 'out_ground']:
        setattr(args, k, getattr(args, k).format(out_dir=args.out_dir, split_name='{split_name}'))
    run(**args.__dict__)
