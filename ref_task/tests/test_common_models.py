import time
import string
import pytest

import numpy as np
import torch
from torch import optim

from ulfs import tensor_utils
from ref_task.models import common_models


@pytest.mark.skip()
def test_stochastic_rnn_decoder_fixed_length_rand():
    print('')
    embedding_size = 8
    seq_len = 5
    vocab_size = 7
    # input_size = 3
    batch_size = 4

    input = torch.rand(batch_size, embedding_size)
    decoder = common_models.StochasticRNNDecoderFixedLength(
        # input_size=input_size,
        embedding_size=embedding_size,
        out_seq_len=seq_len,
        vocab_size=vocab_size
    )
    global_idxes = torch.ones(batch_size, dtype=torch.int64).cumsum(dim=-1) - 1
    print('training')
    decoder.train()
    for it in range(10):
        st, utt = decoder(input, global_idxes)
        print('"' + tensor_utils.tensor_to_str(utt[0]) + '"')
    print('')

    print('eval')
    decoder.eval()
    for it in range(10):
        st, utt = decoder(input, global_idxes)
        print('"' + tensor_utils.tensor_to_str(utt[0]) + '"')
    print('')

    print('training')
    decoder.train()
    for it in range(10):
        st, utt = decoder(input, global_idxes)
        print('"' + tensor_utils.tensor_to_str(utt[0]) + '"')
    print('')


@pytest.mark.skip()
def test_stochastic_rnn_decoder_fixed_length_train():
    N = 4
    embedding_size = 64
    vocab_size = 4
    seq_len = 3
    ent_reg = 0.1

    vocab = string.ascii_lowercase
    input_state = torch.rand(N, embedding_size)
    print('input_state', input_state)
    target_tokens = torch.from_numpy(np.random.choice(vocab_size, (seq_len, N), replace=True))
    print('target_tokens', target_tokens)
    print('target', tensor_utils.tensor_to_str(target_tokens[:, 0]))
    decoder = common_models.StochasticRNNDecoderFixedLength(
        # input_size=input_size,
        embedding_size=embedding_size,
        out_seq_len=seq_len,
        vocab_size=vocab_size
    )
    global_idxes = torch.ones(N, dtype=torch.int64).cumsum(dim=-1) - 1
    print('training')
    decoder.train()

    opt = optim.Adam(lr=0.001, params=decoder.parameters())

    episode = 0
    last_print = time.time()
    while True:
        decoder.train()
        st, utts = decoder(input_state, global_idxes=global_idxes)
        correct_chars = (utts == target_tokens).long().sum(dim=0)
        poss_chars = seq_len
        acc = correct_chars.float() / poss_chars
        rewards = acc

        loss = st.calc_loss(rewards) - ent_reg * st.entropy

        opt.zero_grad()
        loss.backward()
        opt.step()

        if time.time() - last_print >= 3.0:
            print('e', episode)
            print('targets')
            print(tensor_utils.tensor_to_2dstr(target_tokens[:, :4], vocab=vocab))
            # print('targets[0]', tensor_utils.tensor_to_str(target_tokens[0]))
            print('utts')
            print(tensor_utils.tensor_to_2dstr(utts[:, :4], vocab=vocab))
            # print('utts[0]', tensor_utils.tensor_to_str(utts[0]))
            print('acc[:4]', acc[:4])
            print('acc.mean()', acc.mean().item())
            print('loss', loss.item())

            decoder.eval()
            _, eval_utt = decoder(input_state, global_idxes=global_idxes)
            eval_correct_chars = (eval_utt == target_tokens).long().sum(dim=0)
            eval_acc = eval_correct_chars.float() / poss_chars
            eval_acc = eval_acc.mean().item()
            print('eval')
            print(tensor_utils.tensor_to_2dstr(eval_utt[:, :4], vocab=vocab))
            print('eval acc', eval_acc)
            if eval_acc == 1.0:
                print('reached eval acc 1.0, finishing')
                break
            # print('eval[0]', tensor_utils.tensor_to_str(eval_utt[0]))
            last_print = time.time()

        episode += 1


def test_attentional_planar_remapping():
    embedding_size = 32
    N = 4
    C = 7
    H = 5
    W = 5

    images = torch.rand(N, C, H, W)
    atts = torch.rand(N, embedding_size)

    mapping_layer = common_models.AttentionalPlanarRemapping(
        embedding_size=embedding_size, num_channels=C)
    out = mapping_layer(images=images, atts=atts)
    # print('images[0]', images[0])
    # print('out[0]', out[0])
    print('images.size()', images.size())
    print('out.size()', out.size())
