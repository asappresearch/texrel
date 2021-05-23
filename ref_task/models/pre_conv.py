"""
conv layer (or anything really) applied to all images (both sender and receiver) before going
into more specialist models

Baseline is going to apply a stride 2 kernel 2 conv, with 32 planes, and some dropout on the output

This will convert a texture/color input, with 2x2 textures into a grid sized output, with 32 planes
(which might approximately resemble the original one-hot encoded earlier toy data format)
"""
import abc
import torch
from torch import nn
import torch.nn.functional as F

from ulfs import utils


class PreConv(torch.nn.Module, abc.ABC):
    def __call__(self, images: torch.Tensor) -> torch.Tensor:
        return super().__call__(images)

    @abc.abstractmethod
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        pass

    @abc.abstractmethod
    def get_output_size(self, input_size: int) -> int:
        pass

    @abc.abstractmethod
    def get_output_planes(self, input_planes: int) -> int:
        pass


class Identity(PreConv):
    def __init__(self):
        super().__init__()

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return images

    def get_output_size(self, input_size: int) -> int:
        return input_size

    def get_output_planes(self, input_planes: int) -> int:
        return input_planes


class StridedConv(PreConv):
    """
    this will set stride and kernel size to same amount, so shrinks the image
    by a factor of exactly stride (we dont add padding)
    note that image size going in should ideally be an exact multiple of stride
    """
    def __init__(self, input_channels, output_channels, dropout, stride, use_relu):
        super().__init__()
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.stride = stride
        self.use_relu = use_relu
        self.conv1 = nn.Conv2d(
            input_channels, output_channels, kernel_size=stride, stride=stride, padding=0)
        self.drop = nn.Dropout(dropout)

    def get_output_size(self, input_size):
        return input_size // self.stride

    def get_output_planes(self, input_planes):
        return self.output_channels

    def forward(self, images):
        """
        images is assumed to be [...][3][H][W]
        where [...] is any number of dimensions, of any size
        """
        shape_all = list(images.size())
        pre_shape = shape_all[:-3]
        C, H, W = shape_all[-3:]
        assert C == self.input_channels

        images = images.view(-1, C, H, W)
        images = self.conv1(images)
        H //= self.stride
        W //= self.stride
        images = self.drop(images)
        if self.use_relu:
            images = F.relu(images)
        images = images.view(*pre_shape, self.output_channels, H, W)
        return images


def build_preconv(params, ds_meta):
    p = params
    pre_conv = build_preconv_(
        input_channels=ds_meta.grid_planes,
        **utils.filter_dict_by_prefix(p.__dict__, prefix='preconv_', truncate_prefix=True))
    return pre_conv


def build_preconv_(
        model, relu, dropout, embedding_size, input_channels, stride) -> PreConv:
    PreConvClass = globals()[model]
    pre_conv_args = {}
    if PreConvClass == StridedConv:
        pre_conv_args['input_channels'] = input_channels
        pre_conv_args['output_channels'] = embedding_size
        pre_conv_args['dropout'] = dropout
        pre_conv_args['stride'] = stride
        pre_conv_args['use_relu'] = relu
    pre_conv: PreConv = PreConvClass(**pre_conv_args)
    return pre_conv
