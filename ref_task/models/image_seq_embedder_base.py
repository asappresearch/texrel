import abc
import torch
from torch import nn


class ImageSeqEmbedder(nn.Module, abc.ABC):
    def __call__(self, images: torch.Tensor) -> torch.Tensor:
        return super().__call__(images=images)

    @abc.abstractmethod
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        pass

    label_aware = False
