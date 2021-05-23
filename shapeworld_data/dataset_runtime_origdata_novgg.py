"""
forked from reprod/learning_latent_language/dataset.py 2021 apr 16

this is going to use the original dataset (not somethign I generated myself,
cf dataset_runtime.py, and it's going to use the non-featurized data, i.e.
the [3][64][64] images. We'll then feed this through a conv4. In this version.
"""
import argparse
from os.path import join, expanduser as expand
import json

import numpy as np

import torch


class Vocab(object):
    def __init__(self, w2i, vocab):
        self.w2i = w2i
        self.vocab = vocab
        self.vocab_size = len(w2i)


class SplitDataset(object):
    def __init__(self, data_folder, set_name, vocab_obj=None):
        self.labels = torch.from_numpy(np.load(join(data_folder, set_name, 'labels.npz'))['arr_0']).long()
        self.inputs = torch.from_numpy(np.load(join(data_folder, set_name, 'inputs.npz'))['arr_0']).transpose(
            -1, -2).transpose(-2, -3).float().contiguous()
        self.examples = torch.from_numpy(np.load(
            join(data_folder, set_name, 'examples.npz'))['arr_0']).transpose(
                0, 1).transpose(-1, -2).transpose(-2, -3).float().contiguous()
        with open(join(data_folder, set_name, 'hints.json'), 'r') as f:
            c = f.read()
        self.captions = json.loads(c)
        if vocab_obj is not None:
            self.w2i = vocab_obj.w2i
            self.vocab = vocab_obj.vocab
            self.vocab_size = len(self.w2i)
        else:
            self.w2i = {}
            self.vocab = []
            self.w2i['.'] = 0
            self.vocab.append('.')
            self.vocab_size = len(self.w2i)
        self.process_vocab(self.captions)
        print('--------')
        self.N = self.labels.view(-1).size(0)
        print('dataset', set_name, 'N', self.N)
        self.device = 'cpu'
        self.image_size = self.inputs.size(-1)

    def cuda(self):
        self.device = 'cuda'
        return self

    @property
    def vocab_obj(self):
        return Vocab(w2i=self.w2i, vocab=self.vocab)

    def process_vocab(self, captions):
        w2i = self.w2i
        vocab = self.vocab
        max_len = 0
        N = len(captions)
        for c in captions:
            length = len(c.split(' '))
            max_len = max(max_len, length)
        print('max_len', max_len)
        encoded = torch.LongTensor(max_len, N).zero_()
        # mask will be 1 wherever there is a letter, including
        # the first null terminator
        # (0 everywhere else)
        mask = torch.zeros((max_len, N), dtype=bool)
        for n, c in enumerate(captions):
            if n <= 5:
                print('n', n, c)
            for i, w in enumerate(c.split(' ')):
                if w not in w2i:
                    w2i[w] = len(w2i)
                    vocab.append(w)
                encoded[i, n] = w2i[w]
                mask[i, n] = 1
        print('len(w2i)', len(w2i))
        self.w2i = w2i
        self.vocab = vocab
        self.encoded = encoded
        self.encoded_mask = mask
        self.vocab_size = len(vocab)
        print('processed vocab', encoded.size())

    def _batch_from_idxes(self, idxes, no_sender: bool = False):
        labels = self.labels[idxes].to(self.device)
        inputs = self.inputs[idxes].to(self.device)
        captions = [self.captions[idx] for idx in idxes]
        examples = self.examples[:, idxes].to(self.device)
        utterances = self.encoded[:, idxes].to(self.device)
        utterances_mask = self.encoded_mask[:, idxes].to(self.device)
        M, N, _, _, _ = examples.size()
        train_labels = torch.full((M, N), 1, dtype=torch.int64, device=self.device)
        return {
            'inner_test_labels_t': labels.unsqueeze(0),
            'inner_test_examples_t': inputs.unsqueeze(0),
            'inner_train_examples_t': examples,
            'inner_train_labels_t': train_labels,
            'hypotheses_t': utterances,
            'hypotheses_mask_t': utterances_mask,
            'hypotheses_english': captions
        }

    def sample(self, batch_size, no_sender: bool = False):
        idxes = torch.from_numpy(np.random.choice(self.N, batch_size, replace=False)).long()
        return self._batch_from_idxes(idxes=idxes)


class Dataset(object):
    def __init__(self, data_folder, vocab_obj=None):
        data_folder = expand(data_folder)
        self.data_by_split_name = {}
        self.data_by_split_name['train'] = SplitDataset(
            data_folder=data_folder,
            vocab_obj=vocab_obj,
            set_name='train'
        )
        self.data_by_split_name['val_same'] = SplitDataset(
            data_folder=data_folder,
            vocab_obj=vocab_obj,
            set_name='val_same'
        )
        self.data_by_split_name['val_new'] = SplitDataset(
            data_folder=data_folder,
            vocab_obj=vocab_obj,
            set_name='val'
        )
        self.data_by_split_name['test_same'] = SplitDataset(
            data_folder=data_folder,
            vocab_obj=vocab_obj,
            set_name='test_same'
        )
        self.data_by_split_name['test_new'] = SplitDataset(
            data_folder=data_folder,
            vocab_obj=vocab_obj,
            set_name='test'
        )
        self.meta = argparse.Namespace()
        self.meta.grid_planes = 3
        self.meta.grid_size = 64

    def cuda(self):
        self.train_data.cuda()
        self.test_data.cuda()
        return self

    def holdout_iterator(self, batch_size, split_name: str):
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
                    return self.parent._batch_from_idxes(idxes=idxes, no_sender=False)
                else:
                    raise StopIteration

        data = self.data_by_split_name[split_name]
        N = data.N
        num_batches = N // batch_size
        return Iterator(parent=data)

    def sample(self, batch_size: int, split_name: str, no_sender: bool = False):
        data = self.data_by_split_name[split_name]
        return data.sample(batch_size=batch_size)


def run(args):
    dataset = Dataset(data_folder=expand('~/data/shapeworld'), set_name='test')
    batch = dataset.sample(batch_size=args.num_samples)
    for n in range(args.num_samples):
        print('')
        print('=====================')
        print('n', n)
        print('test_labels', n, batch['test_labels_t'][:, n])
        print('train_labels', n, batch['train_labels_t'][:, n])
        utterance_txt = ' '.join([dataset.vocab[idx] for idx in batch['utterances'][:, n]])
        print('utterance', utterance_txt)
        test_example = batch['test_examples_t'][0, n]
        print('test_example.size()', test_example.size())
        print('train_examples_t.size()', batch['train_examples_t'].size())


if __name__ == '__main__':
    # for testing really
    parser = argparse.ArgumentParser()
    parser.add_argument('-N', '--num-samples', type=int, default=5)
    args = parser.parse_args()
    run(args)
