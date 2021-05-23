import torch
from torch import nn

from ulfs import tensor_utils
from ref_task.models import (
    pre_conv as pre_conv_lib,
    linguistic_encoders,
    multimodal_classifiers)


class ReceiverModel(nn.Module):
    """
    takes a relation/utterance/hypothesis that describe some kind of
    relationship or similar, and outputs whether it is consistent with
    the receiver_image (operates on mini-batches)
    - embeds the utterances using a linguistic encoder
    - embeds the images using a pre_conv model
    - uses a multimodal_classifier to combine the two

    output is logits

    # output is probability distributions, ie already softmaxed
    # when using gumbel or REINFORCE, output is one-hot probaiblity dsitributions

    # for reinforce, we'd need to either implement calc_loss, to call self.sample.calc_loss,
    # or expose the stochastic trajectory, or something else (ie reinforce not currently
    # implemented with this)
    """
    def __init__(
            self,
            pre_conv: pre_conv_lib.PreConv,
            multimodal_classifier: multimodal_classifiers.MultimodalClassifier,
            linguistic_encoder: linguistic_encoders.LinguisticEncoder):
        super().__init__()
        self.pre_conv = pre_conv
        self.linguistic_encoder = linguistic_encoder
        self.multimodal_classifier = multimodal_classifier

    def __call__(self, utts: torch.Tensor, images: torch.Tensor) -> torch.Tensor:
        return super().__call__(utts=utts, images=images)

    def forward(self, utts: torch.Tensor, images: torch.Tensor) -> torch.Tensor:
        images = self.pre_conv(images=images)
        utts_emb = self.linguistic_encoder(utts=utts)

        if len(images.size()) == 5:
            M = images.size(0)
            merger = tensor_utils.DimMerger()
            images_flat = merger.merge(images.contiguous(), 0, 1)
            utts_emb_size = list(utts_emb.size())
            utts_emb_expanded = utts_emb.unsqueeze(0).expand(
                 M, *utts_emb_size).contiguous()
            utts_emb_flat = merger.merge(utts_emb_expanded, 0, 1)
            out_logits_flat = self.multimodal_classifier(
                utts_emb=utts_emb_flat, images=images_flat)
            out_logits = merger.resplit(out_logits_flat, 0)
        else:
            out_logits = self.multimodal_classifier(utts_emb=utts_emb, images=images)

        return out_logits


def build_receiver_model(params, ds_meta, utt_len: int, vocab_size: int, pre_conv=None) -> ReceiverModel:
    """
    given the size of images from a dataset, and a desired vocab size and utterance length,
    creates a ReceiverModel, which will take in images, and utterances, and classify
    the images as being consistent with the utterances or not.
    """
    p = params
    if pre_conv is None:
        pre_conv = pre_conv_lib.build_preconv(params=p, ds_meta=ds_meta)

    multimodal_classifier = multimodal_classifiers.build_multimodal_classifier(
        params=p, pre_conv=pre_conv, ds_meta=ds_meta)
    linguistic_encoder = linguistic_encoders.build_linguistic_encoder(
        params=p, utt_len=utt_len, vocab_size=vocab_size)

    receiver_model = ReceiverModel(
        pre_conv=pre_conv,
        multimodal_classifier=multimodal_classifier,
        linguistic_encoder=linguistic_encoder)
    return receiver_model
