from typing import Iterable, Optional, List
from torch import nn
# import abc
import torch
# from ref_task.models import common_models
# from ref_task.models import pre_conv as pre_conv_lib


class CNNModel(nn.Module):
    def __init__(
            self, grid_planes: int, cnn_sizes: Iterable[int],
            batch_norm: bool, max_pooling_size: Optional[int]) -> None:
        super().__init__()
        self.cnn_sizes = cnn_sizes
        last_channels = grid_planes
        cnn_blocks = []
        for i, cnn_size in enumerate(cnn_sizes):
            block: List[nn.Module] = []
            # if i != 0:
            #     block.append(nn.Dropout(dropout))
            #     block.append(nn.ReLU())
            block.append(
                nn.Conv2d(in_channels=last_channels, out_channels=cnn_size, kernel_size=3, padding=1))
            if batch_norm:
                block.append(nn.BatchNorm2d(cnn_size))
            block.append(nn.ReLU())
            if max_pooling_size is not None:
                block.append(nn.MaxPool2d(kernel_size=max_pooling_size))
            last_channels = cnn_size
            cnn_blocks.append(nn.Sequential(*block))
        self.conv = nn.Sequential(*cnn_blocks)

    def __call__(self, images: torch.Tensor) -> torch.Tensor:
        return super().__call__(images=images)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        images = self.conv(images)
        return images


# class ConvModelBase(nn.Module, abc.ABC):
#     def __call__(self, images: torch.Tensor) -> torch.Tensor:
#         return super().__call__(images=images)

#     @abc.abstractmethod
#     def forward(self, images: torch.Tensor) -> torch.Tensor:
#         pass


# class StridedPlusModel(ConvModelBase):
#     def __init__(self, params, ds_meta):
#         super().__init__()
#         self.pre_conv = pre_conv_lib.build_preconv(params=params, ds_meta=ds_meta)
#         self.conv = common_models.CNNModel(
#             grid_planes=ds_meta.grid_planes,
#             cnn_sizes=params.cnn_sizes,
#             dropout=params.dropout)

#     def forward(self, images):
#         images = self.pre_conv(images)
#         images = self.conv(images)
#         return images


# class Conv4Model(ConvModelBase):
#     def __init__(self, params, ds_meta, num_blocks: int = 4, channels: int = 64, kernel_size: int = 3):
#         # the model from prototypical paper
#         # 4 blocks:
#         # - 64 filter 3x3
#         # - batch normalization
#         # - ReLU
#         # - 2x2 max pooling
#         super().__init__()
#         blocks_l = []
#         in_channels = ds_meta.grid_planes
#         for b in range(num_blocks):
#             block = nn.Sequential(*[
#                 nn.Conv2d(in_channels=in_channels, out_channels=channels, kernel_size=3, padding=1),
#                 nn.BatchNorm2d(num_features=channels),
#                 nn.ReLU(),
#                 nn.MaxPool2d(kernel_size=2)
#             ])
#             in_channels = channels
#             blocks_l.append(block)
#         self.conv = nn.Sequential(*blocks_l)

#     def forward(self, images):
#         images = self.conv(images)
#         return images


# def build_conv_model(conv_model: str, params, ds_meta) -> ConvModelBase:
#     ConvModelClass = globals()['f{conv_model}Model']
#     conv_model_object = ConvModelClass(params=params, ds_meta=ds_meta)
#     return conv_model_object
