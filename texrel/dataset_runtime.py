"""
handles sampling data from ondisk file at runtime
"""
from typing import List, Dict, Any, Optional
from collections import defaultdict
from os.path import expanduser as expand
import time
import pickle
import gzip
import itertools

import numpy as np
import torch

from ulfs.params import Params

from texrel.texturizer import Texturizer


class TexRelDataset(object):
    def __init__(
        self,
        ds_filepath_templ: str,
        ds_refs: List[str],
        ds_seed: int,
        ds_texture_size: int,
        ds_background_noise: float,
        ds_mean: float,
        ds_mean_std: float,
        ds_val_refs: Optional[List[str]] = None,
    ):
        """
        ds_filepath_templ: str
            filepath for each data file, templatized with `{ds_ref}`, e.g.
            "~/data/texrel/{ds_ref}.dat"
        ds_refs: List[str]
            List of ds_refs of the datafiles we want to load
        ds_seed: int
            seed used for initializing the textures and colors
        ds_texture_size: int
            how big to make each texture, each texture will have
            its width and height set to ds_texture_size
        ds_background_noise: float
            how much gaussian noise to add to the background colors of each
            generated image. this noise is applied per-pixel
        ds_mean: float
            the mean brightness of the background, [0-1]
        ds_mean_std: float
            standard deviation of noise. this is applied over entire images
            (cf ds_background_noise, which is per-pixel)
        ds_val_refs: Optional[List[str]] = None
            List of ds_refs to use for validation and test sets. If None,
            then the ones in `ds_refs` will be used for validation and test
        """
        print('ds-refs', ds_refs)
        print('ds-val-refs', ds_val_refs)
        texture_size = ds_texture_size
        background_noise = ds_background_noise
        self.ds_mean_std = ds_mean_std

        self.background_noise = background_noise

        self.metas: List[Any] = []
        dsref_name2id: Dict[str, int] = {}
        self.datas_by_dsref = {}  # eg {'ds63': {'train': {'N': ..., 'input_shapes': ..., ...}}}
        self.datas: Dict[str, Dict[str, Any]] = {}  # {'train': {'N': ..., 'input_shapes': ..., ...}}

        if ds_val_refs is None:
            ds_val_refs = list(ds_refs)
        all_ds_refs = list(set(ds_refs) | set(ds_val_refs))
        for ds_ref in all_ds_refs:
            """
            ds_ref, eg dsref64
            """
            dsref_name2id[ds_ref] = len(dsref_name2id)
            print(f'loading {ds_ref} ...', end='', flush=True)
            start_time = time.time()
            filepath = ds_filepath_templ.format(ds_ref=ds_ref)
            with gzip.open(expand(filepath), 'rb') as f:
                d = pickle.load(f)  # type: ignore
            _meta = d['meta']
            load_time = time.time() - start_time
            print(f' done in {load_time:.1f}s')
            version = _meta.get('version', 'v1')
            print('    ', 'data format version', version)
            self.metas.append(Params(d['meta']))
            self.datas_by_dsref[ds_ref] = d['data']
            # split_name is eg 'train', 'holdout'
            for split_name, data in d['data'].items():
                _ds_refs = ds_refs if split_name == 'train' else ds_val_refs
                if ds_ref not in _ds_refs:
                    continue
                print('    ', split_name, ds_ref, end='', flush=True)
                _N = data['inner_train_labels'].shape[1]
                print(' N', _N)
                if split_name not in self.datas:
                    self.datas[split_name] = defaultdict(list)
                for k2, v in data.items():
                    """
                    k2, eg 'N', 'input_labels', 'input_shapes', ...
                    """
                    self.datas[split_name][k2].append(v)
                dsrefs_t = torch.full((_N, ), fill_value=dsref_name2id[ds_ref], dtype=torch.int64)
                self.datas[split_name]['dsrefs_t'].append(dsrefs_t)
        datas_new: Dict[str, Dict[str, Any]] = {}
        for split_name, data in self.datas.items():
            datas_new[split_name] = {}
            d = datas_new[split_name]
            d['N'] = np.sum(data['N']).item()
            tensor_dim_by_name = {
                'inner_train_labels': 1,
                'inner_train_shapes': 1,
                'inner_train_colors': 1,
                'inner_test_shapes': 1,
                'inner_test_colors': 1,
                'inner_test_labels': 1,
                'ground_colors': 1,
                'ground_shapes': 1,
                'dsrefs_t': 0
            }
            for name, dim in tensor_dim_by_name.items():
                if len(ds_refs) == 1 or name not in 'hypotheses_t':
                    _v_l = []
                    for v in data[name]:
                        if isinstance(v, np.ndarray):
                            v = torch.from_numpy(v)
                        _v_l.append(v)
                    v = torch.cat(_v_l, dim=dim)
                    d[name] = v
                else:
                    _max_utt_len = 0
                    _N = 0
                    for t in data[name]:
                        _max_utt_len = max(_max_utt_len, t.shape[0])
                        _N += t.size(1)
                    _fused_shape = list(t.size())
                    _fused_shape[0] = _max_utt_len
                    _fused_shape[1] = _N
                    v = torch.zeros(*_fused_shape, dtype=torch.int64)
                    _n = 0
                    for t in data[name]:
                        v[:t.size(0), _n:_n + t.size(1)] = t
                        _n += t.size(1)
                    d[name] = v
            for k2 in ['hypotheses_english', 'hypotheses_structured']:
                if k2 in data:
                    datas_new[split_name][k2] = list(itertools.chain.from_iterable(data[k2]))
                else:
                    print('warning, not found:', k2)

        overall_meta: Dict[str, Any] = {}
        print('')
        for meta in self.metas:
            print('meta')
            print(meta)
            for k, v in meta.__dict__.items():
                if k not in ['ref', 'ds_ref', 'hypothesis_generators', 'seed', 'num_distractors', 'num_holdout']:
                    if k in overall_meta:
                        if overall_meta[k] != v:
                            print(f'meta mismatch {k}: {overall_meta[k]} != {v}')
                        assert overall_meta[k] == v
                    overall_meta[k] = v
        overall_meta: Any = Params(overall_meta)
        self.meta = overall_meta
        print('overall meta', self.meta)

        words_set = set()
        max_seq_len = 0
        for split_name, data in self.datas.items():
            _words = set(itertools.chain.from_iterable(data['words']))
            max_seq_len = max(max_seq_len, max(data['hypothesis_english_max_len']))
            words_set |= _words
        words = sorted(list(words_set))
        print('words', words)
        print('max_seq_len', max_seq_len)
        self.i2w = words
        self.w2i = {w: i for i, w in enumerate(words)}
        self.max_seq_len = max_seq_len

        self.datas = datas_new
        if 'test' in self.datas:
            self.datas['val_new'] = self.datas['test']
            del self.datas['test']

        self.meta.grid_planes = 3
        self.meta.grid_size *= texture_size
        self.meta.vocab_size = len(self.i2w)
        self.meta.utt_len = max_seq_len
        self.meta.M_train = self.metas[0].inner_train_pos + self.metas[0].inner_train_neg
        self.meta.M_test = self.metas[0].inner_test_pos + self.metas[0].inner_test_neg
        print('M_train', self.meta.M_train, 'M_test', self.meta.M_test)

        self.training = True
        self.texturizer = Texturizer(
            num_textures=self.meta.num_shapes,
            num_colors=self.meta.num_colors,
            texture_size=ds_texture_size,
            seed=ds_seed,
            background_noise=background_noise,
            background_mean_std=ds_mean_std,
            background_mean=ds_mean
        )
        self.device = 'cpu'

    @classmethod
    def from_collection(
        cls,
        ds_filepath_templ: str,
        ds_seed: int,
        ds_texture_size: int,
        ds_background_noise: float,
        ds_mean: float,
        ds_mean_std: float,
        ds_collection: str,
        ds_tasks: List[str],
        ds_distractors: List[int],
        ds_val_tasks: Optional[List[str]] = None,
        ds_val_distractors: Optional[List[int]] = None,
    ):
        """
        Parameters
        ----------
        ds_filepath_templ: str
            filepath for each data file, templatized with `{ds_ref}`, e.g.
            "~/data/texrel/{ds_ref}.dat"
        ds_seed: int
            seed used for initializing the textures and colors
        ds_texture_size: int
            how big to make each texture, each texture will have
            its width and height set to ds_texture_size
        ds_background_noise: float
            how much gaussian noise to add to the background colors of each
            generated image. this noise is applied per-pixel
        ds_mean: float
            the mean brightness of the background, [0-1]
        ds_mean_std: float
            standard deviation of noise. this is applied over entire images
            (cf ds_background_noise, which is per-pixel)
        ds_collection: str
            reference name for collection, as passed to create_collection.py script
        ds_tasks: List[str]
            list of tasks, eg ['Colors1', 'Relations'], for train
        ds_distractors: List[int]
            list of distractor counts, eg [0, 2], for train
        ds_val_tasks: Optional[List[str]]
            tasks for validation and test; if None, then ds_tasks are used
        ds_distractors: Optional[List[int]]
            distractors for validation and test; if None then ds_distractors are used
        """
        ds_refs = []
        if ds_val_tasks is None:
            ds_val_tasks = list(ds_tasks)

        if ds_val_distractors is None:
            ds_val_distractors = list(ds_distractors)

        def tasks_dists_to_refs(ds_tasks, ds_distractors):
            ds_refs = []
            for ds_task in ds_tasks:
                for ds_dist in ds_distractors:
                    ds_ref = f'{ds_collection}-{ds_task.lower()}-d{ds_dist}'
                    ds_refs.append(ds_ref)
            return ds_refs

        ds_refs = tasks_dists_to_refs(ds_tasks=ds_tasks, ds_distractors=ds_distractors)
        ds_val_refs = tasks_dists_to_refs(ds_tasks=ds_val_tasks, ds_distractors=ds_val_distractors)
        return cls(
            ds_filepath_templ=ds_filepath_templ,
            ds_seed=ds_seed,
            ds_texture_size=ds_texture_size,
            ds_background_noise=ds_background_noise,
            ds_mean=ds_mean,
            ds_mean_std=ds_mean_std,
            ds_refs=ds_refs,
            ds_val_refs=ds_val_refs
        )

    def summarize_datas(self):
        for name, datas in self.datas.items():
            print(f'{name}:')
            for k, v in datas.items():
                if isinstance(v, int):
                    print('  ', k, v)
                else:
                    print('  ', k, v.dtype, v.size())

    def summarize_datas_by_dsref(self):
        for dsref, data in self.datas_by_dsref.items():
            print(f'{dsref}:')
            for name, datas in data.items():
                print(f'  {name}:')
                for k, v in datas.items():
                    if isinstance(v, int):
                        print('    ', k, v)
                    else:
                        print('    ', k, v.dtype, v.size())

    def eval(self):
        self.training = False
        return self

    def train(self):
        self.training = True
        return self

    def cuda(self):
        self.device = 'cuda'
        self.texturizer.cuda()
        return self

    @property
    def N(self):
        if self.training:
            return self.datas['train']['N']
        return self.datas['val_new']['N']

    def _batch_from_idxes(self, split_name: str, idxes: torch.Tensor, no_sender: bool, add_ground: bool):
        data = self.datas[split_name]
        N = len(idxes)

        hypotheses_english = [data['hypotheses_english'][idx] for idx in idxes]
        hypotheses_structured = [data['hypotheses_structured'][idx] for idx in idxes]

        hypotheses_t = torch.zeros((self.max_seq_len, N), dtype=torch.int64)
        for n, hyp_eng in enumerate(hypotheses_english):
            _hyp_t = torch.LongTensor([self.w2i[word] for word in hyp_eng.split()])
            _l = _hyp_t.size(0)
            hypotheses_t[:_l, n] = _hyp_t

        inner_test_shapes_t = data['inner_test_shapes'][:, idxes].long().to(self.device)
        inner_test_colors_t = data['inner_test_colors'][:, idxes].long().to(self.device)
        inner_test_labels_t = data['inner_test_labels'][:, idxes].long().to(self.device)

        if add_ground:
            ground_colors_t = data['ground_colors'][:, idxes].long().to(self.device)
            ground_shapes_t = data['ground_shapes'][:, idxes].long().to(self.device)
            ground_examples_t = self.texturizer.forward(
                texture_idxes=ground_shapes_t, color_idxes=ground_colors_t)

        if not no_sender:
            inner_train_shapes_t = data['inner_train_shapes'][:, idxes].long().to(self.device)
            inner_train_colors_t = data['inner_train_colors'][:, idxes].long().to(self.device)
            inner_train_labels_t = data['inner_train_labels'][:, idxes].long().to(self.device)
            inner_train_examples_t = self.texturizer.forward(
                texture_idxes=inner_train_shapes_t, color_idxes=inner_train_colors_t)

        inner_test_examples_t = self.texturizer.forward(
            texture_idxes=inner_test_shapes_t, color_idxes=inner_test_colors_t)

        dsrefs_t = data['dsrefs_t'][idxes].long().to(self.device)

        res = {
            'N': N,
            'inner_test_shapes_t': inner_test_shapes_t,
            'inner_test_colors_t': inner_test_colors_t,
            'inner_test_examples_t': inner_test_examples_t.detach(),
            'inner_test_labels_t': inner_test_labels_t.detach(),
            'dsrefs_t': dsrefs_t.detach(),
            'hypotheses_t': hypotheses_t,
            'hypotheses_english': hypotheses_english,
            'hypotheses_structured': hypotheses_structured,
        }
        if add_ground:
            res['ground_examples_t'] = ground_examples_t
        if not no_sender:
            res['inner_train_examples_t'] = inner_train_examples_t.detach()
            res['inner_train_labels_t'] = inner_train_labels_t.detach()
            res['inner_train_colors_t'] = inner_train_colors_t.detach()
            res['inner_train_shapes_t'] = inner_train_shapes_t.detach()

        return res

    def holdout_iterator(self, batch_size: int, split_name: str):
        class Iterator(object):
            def __init__(self, parent):
                self.b = 0
                self.parent = parent

            def __iter__(self):
                return self

            def __next__(self):
                if self.b < num_batches:
                    idxes = torch.arange(self.b * batch_size, (self.b + 1) * batch_size, dtype=torch.int64)
                    self.b += 1
                    return self.parent._batch_from_idxes(
                        split_name=split_name, idxes=idxes, no_sender=False, add_ground=False)
                else:
                    raise StopIteration

        data = self.datas[split_name]
        N = data['N']
        num_batches = N // batch_size
        return Iterator(parent=self)

    def sample(
            self, batch_size: int, split_name: str, no_sender: bool = False,
            add_ground: bool = False):
        data = self.datas[split_name]
        N = data['N']
        idxes = torch.from_numpy(np.random.choice(N, batch_size, replace=False))
        return self._batch_from_idxes(split_name=split_name, idxes=idxes, no_sender=no_sender, add_ground=add_ground)
