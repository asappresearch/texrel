import argparse
import torch

from ref_task.models import receiver_model as receiver_model_lib


def test_receiver_model(params: argparse.Namespace):
    utt_len = 6
    vocab_size = 10
    batch_size = 5
    grid_planes = 3
    grid_size = 12

    ds_meta = argparse.Namespace()
    ds_meta.grid_planes = grid_planes
    ds_meta.grid_size = grid_size

    receiver_model = receiver_model_lib.build_receiver_model(
        params, ds_meta, utt_len=utt_len, vocab_size=vocab_size, pre_conv=None
    )
    print('receiver_model', receiver_model)
    utts = torch.rand(utt_len, batch_size, vocab_size)
    receiver_images = torch.rand(batch_size, grid_planes, grid_size, grid_size)
    output = receiver_model(utts=utts, images=receiver_images)
    print('output', output)
    print('output.size()', output.size())
