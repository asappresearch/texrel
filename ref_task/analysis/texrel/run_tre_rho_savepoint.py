"""
given a save of utterances hypotheses etc, calculate rho and tre
"""
import time
import argparse
from os.path import expanduser as expand

import torch

from ulfs import metrics as metrics_lib, tre as tre_lib, tensor_utils


def run(args):
    with open(args.in_filepath, 'rb') as f:
        d = torch.load(f, map_location=torch.device('cpu'))
    print(d.keys())

    hypotheses_t, hypotheses_structures, tokens_t = map(d.__getitem__, [
        'hypotheses_t', 'hypotheses_structures', 'tokens_t'])

    print('hypotheses_t.size()', hypotheses_t.size(), hypotheses_t.dtype, hypotheses_t.max())
    print('tokens_t.size()', tokens_t.size(), tokens_t.dtype, tokens_t.max())

    before_rho = time.time()
    rho = metrics_lib.topographic_similarity(
        utts=tokens_t, labels=hypotheses_t
    )
    rho_time = time.time() - before_rho
    print('rho=%.3f' % rho, 't=%.3f' % rho_time)

    N, S = tokens_t.size()

    for n in range(5):
        print('n', tokens_t[n])

    before_tre = time.time()
    tokens_onehot = tensor_utils.idxes_to_onehot(idxes=tokens_t, vocab_size=args.vocab_size)
    tokens_onehot = tensor_utils.merge_dims(tokens_onehot, -2, -1)
    print('tokens_onehot.size()', tokens_onehot.size())
    Compose = getattr(tre_lib, f'{args.compose_model}Compose')
    comp_fn = Compose(
        num_terms=2,
        vocab_size=args.vocab_size,
        msg_len=S,
        bias=args.bias,
    )
    tre = tre_lib.evaluate(
        reps=tokens_onehot[:args.tre_samples],
        oracle_structures=hypotheses_structures[:args.tre_samples],
        comp_fn=comp_fn,
        distance_fn=tre_lib.L1Dist(),
        steps=args.tre_steps,
        tre_lr=args.tre_lr,
    )
    tre_time = time.time() - before_tre
    print('tre=%.3f' % tre, 't=%.3f' % tre_time)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--in-filepath', type=str, default=expand('~/data/tre/hyps_451.pt'))
    parser.add_argument('--compose-model', type=str, default='ProjectionSum')
    parser.add_argument('--vocab-size', type=int, default=26)
    parser.add_argument('--tre-samples', type=int, default=80)
    parser.add_argument('--tre-steps', type=int, default=400)
    parser.add_argument('--tre-lr', type=float, default=0.01)
    parser.add_argument('--bias', action='store_true', help='add bias to composition')
    args = parser.parse_args()
    run(args)
