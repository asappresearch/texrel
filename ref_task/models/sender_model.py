from typing import Union
import torch
from torch import nn

from ref_task.models import (
    image_seq_embedders, sender_model as sender_model_lib, decoders,
    pre_conv as pre_conv_lib)


class SenderModel(nn.Module):
    """
    takes a sequence of images, and labels for each image (positive/negative), and outputs
    a hypothesis/utterance which explains those images
    - simply adds the labels as an additional image plane, into each image stack (along with
      r, g, b planes for example)

    returns logits (or something else if using reinforce)
    """
    def __init__(
            self,
            pre_conv: pre_conv_lib.PreConv,
            image_seq_embedder,
            decoder,
            enable_negative_examples: bool):
        """
        If labels are provided in forward, these are combined into the convolved images as additional
        image planes, assuming that image_seq_embedder.image_aware = False. But if image_aware is True,
        then the labels are passed to the image_seq_embedder as an additional parameter.

        sender handles decoding unless image_seq_embedder.handles_decoding is True

        parameters
        ==========
        - pre_conv: first convolutional layer
        - image_seq_embedder: a model which takes convolved images
          and converts them into an embedding
        - decoder: a linguistic decoder, which takes an embedding and outputs an utterance
        """
        super().__init__()
        self.pre_conv = pre_conv
        self.image_seq_embedder = image_seq_embedder
        self.decoder = decoder
        self.enable_negative_examples = enable_negative_examples

    def forward(self, images: torch.Tensor, labels: torch.Tensor):
        images = self.pre_conv(images)
        if self.enable_negative_examples and not self.image_seq_embedder.label_aware:
            M, N, C, H, W = images.size()
            images_new = torch.zeros(M, N, C + 1, H, W, dtype=torch.float32, device=images.device)
            images_new[:, :, :C] = images
            labels = labels.unsqueeze(-1).unsqueeze(-1).expand(M, N, H, W)
            images_new[:, :, C] = labels
            images = images_new
        if self.image_seq_embedder.label_aware:
            out = self.image_seq_embedder(images=images, labels=labels)
        else:
            out = self.image_seq_embedder(images=images)
        out = self.decoder(out)
        return out


def build_sender_model(params, ds_meta, utt_len: int, vocab_size: int, use_reinforce: bool, pre_conv=None):
    p = params
    if pre_conv is None:
        pre_conv = pre_conv_lib.build_preconv(params=p, ds_meta=ds_meta)

    image_seq_embedder = image_seq_embedders.build_image_seq_embedder(
        params=p, ds_meta=ds_meta, utt_len=utt_len, vocab_size=vocab_size, pre_conv=pre_conv)
    if use_reinforce:
        decoder: Union[
            decoders.DifferentiableDecoder, decoders.StochasticDecoder] = decoders.build_stochastic_decoder(
            params=p, utt_len=utt_len, vocab_size=vocab_size)
    else:
        decoder = decoders.build_differentiable_decoder(
            params=p, utt_len=utt_len, vocab_size=vocab_size)
    sender_model = sender_model_lib.SenderModel(
        pre_conv=pre_conv,
        image_seq_embedder=image_seq_embedder,
        decoder=decoder,
        enable_negative_examples=p.sender_negex
    )
    return sender_model
