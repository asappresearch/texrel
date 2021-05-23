import argparse
import torch
import numpy as np

from ref_task.models import sender_model as sender_model_lib


def test_sender_model(params: argparse.Namespace):
    utt_len = 6
    vocab_size = 10
    batch_size = 5
    grid_planes = 3
    grid_size = 12
    M = 9

    # ds_texture_size = 3
    # embedding_size = 7
    # dropout = 0
    # preconv_dropout = 0.2

    # params = argparse.Namespace()
    # params.ds_texture_size = ds_texture_size
    # params.preconv_relu = False
    # params.preconv_model = 'StridedConv'
    # params.preconv_dropout = preconv_dropout
    # params.preconv_embedding_size = embedding_size
    # params.image_seq_embedder = 'RCNN'
    # params.sender_negex = True
    # params.embedding_size = embedding_size
    # params.dropout = dropout
    # params.cnn_sizes = [4, 4]
    # params.sender_num_rnn_layers = 1
    # params.sender_decoder = 'RNNDecoder'

    ds_meta = argparse.Namespace()
    ds_meta.grid_planes = grid_planes
    ds_meta.grid_size = grid_size

    sender_model = sender_model_lib.build_sender_model(
        params=params, ds_meta=ds_meta, utt_len=utt_len, vocab_size=vocab_size, use_reinforce=False, pre_conv=None)
    print(sender_model)
    images = torch.rand(M, batch_size, grid_planes, grid_size, grid_size)
    labels = torch.from_numpy(np.random.choice(2, (M, batch_size), replace=True))
    outputs = sender_model(images=images, labels=labels)
    print('output.size()', outputs.size())
    assert list(outputs.size()) == [utt_len, batch_size, vocab_size]
