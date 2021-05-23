"""
Inputs to these models are:
- utts_emb: embedded utterances, using a linguistic encoder
- images: images, passed through preconv

Output is, for each pair of examples in utts_emb and images, whether they are consistent
"""
import abc
import torch
from torch import nn
from typing import List, Optional

import numpy as np

from ulfs import tensor_utils, asserts, nn_modules
from ref_task.models import common_models, conv_models, pre_conv as pre_conv_lib


class MultimodalClassifier(nn.Module, abc.ABC):
    def __call__(self, utts_emb: torch.Tensor, images: torch.Tensor) -> torch.Tensor:
        return super().__call__(utts_emb=utts_emb, images=images)

    @abc.abstractmethod
    def forward(self, utts_emb: torch.Tensor, images: torch.Tensor) -> torch.Tensor:
        pass


# class PrototypicalReceiverModel(MultimodalClassifier):
#     """
#     assumes incoming utts_emb is for a single prototype, representing
#     the positive class. compare embedding of image with this prototype
#     and return true if nearer than a certain threshold
#     (based on L3 paper)
#     """
#     def __init__(
#         self,
#         embedding_size: int,
#         dropout: float,
#         cnn_sizes: List[int],
#         grid_planes: int,
#         grid_size: int,
#         cnn_max_pooling_size: Optional[int],
#         cnn_batch_norm: bool,
#     ):
#         super().__init__()
#         self.embedding_size = embedding_size
#         self.dropout = dropout
#         self.cnn_sizes = cnn_sizes
#         self.grid_planes = grid_planes
#         self.grid_size = grid_size

#         self.drop: nn.Dropout = nn.Dropout(dropout)
#         self.cnn_model: conv_models.CNNModel = conv_models.CNNModel(
#             grid_planes=grid_planes,
#             cnn_sizes=cnn_sizes,
#             max_pooling_size=cnn_max_pooling_size,
#             batch_norm=cnn_batch_norm
#         )
#         sample_image = torch.zeros(1, grid_planes, grid_size, grid_size)
#         output_image = self.cnn_model(sample_image)
#         _, C_out, H_out, W_out = output_image.size()
#         self.shrink_cnn_output: nn.Linear = nn.Linear(C_out * H_out * W_out, embedding_size)
#         self.flatten: nn_modules.Flatten = nn_modules.Flatten()

#     def forward(self, utts_emb: torch.Tensor, images: torch.Tensor) -> torch.Tensor:
#         # we need to know how many labels there are...
#         # we assume the two label case, where 1 is positive, and 0
#         # is negative
#         device = utts_emb.device
#         N, E = utts_emb.size()
#         assert N == images.size(0)

#         images = self.cnn_model(images)
#         images = self.drop(images)
#         images = self.flatten(images)
#         images = self.shrink_cnn_output(images)

#         dp = torch.bmm(utts_emb.unsqueeze(1), images.unsqueeze(2)).view(-1)
#         # change this into size 2 tensor... (not sure if this is a good idea really...)
#         dp2 = torch.zeros(N, 2, device=device)
#         dp2[:, 1] = dp
#         dp2[:, 0] = - dp
#         return dp2


class ConcatModel(MultimodalClassifier):
    def __init__(
        self,
        embedding_size: int, cnn_sizes: List[int], grid_planes: int, grid_size: int, dropout: float,
        num_output_fcs: int,
        cnn_max_pooling_size: Optional[int],
        cnn_batch_norm: bool,
    ):
        super().__init__()
        self.embedding_size = embedding_size

        self.drop = nn.Dropout(dropout)
        self.cnn_model = conv_models.CNNModel(
            grid_planes=grid_planes,
            cnn_sizes=cnn_sizes,
            max_pooling_size=cnn_max_pooling_size,
            batch_norm=cnn_batch_norm
        )
        test_input = torch.zeros(1, grid_planes, grid_size, grid_size)
        with torch.no_grad():
            test_output = self.cnn_model(test_input).view(1, -1)
        self.output_model = common_models.OutputFCs(
            input_size=embedding_size + test_output.size(1),
            embedding_size=embedding_size,
            num_output_fcs=num_output_fcs,
            dropout=dropout
        )
        self.flatten = nn_modules.Flatten()

    def forward(self, utts_emb: torch.Tensor, images: torch.Tensor) -> torch.Tensor:
        utts_emb = self.drop(utts_emb)

        images = self.cnn_model(images)
        images = self.drop(images)
        images = self.flatten(images)

        x = torch.cat([utts_emb, images], dim=-1)
        out = self.output_model(x)
        return out


class CosineModel(MultimodalClassifier):
    """
    check if dot product is positive or negative
    classify accordingly
    """
    def __init__(
        self,
        embedding_size: int, cnn_sizes: List[int], grid_planes: int, grid_size: int, dropout: float,
        cnn_max_pooling_size: Optional[int],
        cnn_batch_norm: bool,
    ):
        super().__init__()
        self.embedding_size = embedding_size

        self.drop = nn.Dropout(dropout)
        self.cnn_model = conv_models.CNNModel(
            grid_planes=grid_planes,
            cnn_sizes=cnn_sizes,
            max_pooling_size=cnn_max_pooling_size,
            batch_norm=cnn_batch_norm
        )
        test_input = torch.zeros(1, grid_planes, grid_size, grid_size)
        with torch.no_grad():
            test_output = self.cnn_model(test_input).view(1, -1)
        self.flatten = nn_modules.SpatialToVector()
        self.shrink_cnn_output = nn.Linear(test_output.size(1), embedding_size)

    def forward(self, utts_emb: torch.Tensor, images: torch.Tensor) -> torch.Tensor:
        """
        utts_emb shoudl be [N][E]
        images should be [N][C][H][W]

        output will be [N][2]
        """
        device = utts_emb.device
        N, E = utts_emb.size()
        assert E == self.embedding_size
        assert N == images.size(0)

        images = self.cnn_model(images)
        images = self.drop(images)
        images = self.flatten(images)
        images = self.shrink_cnn_output(images)

        dp = torch.bmm(utts_emb.unsqueeze(1), images.unsqueeze(2)).view(-1)
        # change this into size 2 tensor... (not sure if this is a good idea really...)
        dp2 = torch.zeros(N, 2, device=device)
        dp2[:, 1] = dp
        dp2[:, 0] = - dp
        return dp2


class FeaturePlaneAttentionModel(MultimodalClassifier):
    """
    as per "Gated-Attention Architectures for Task-Oriented Language Grounding"
    """
    def __init__(
        self, embedding_size, cnn_sizes, grid_planes, grid_size, dropout,
        num_output_fcs,
        cnn_max_pooling_size: Optional[int],
        cnn_batch_norm: bool,
        **kwargs
    ):
        asserts.assert_values_empty_none_zero(kwargs)
        super().__init__()
        self.embedding_size = embedding_size
        self.grid_size = grid_size
        self.cnn_sizes = cnn_sizes

        self.drop = nn.Dropout(dropout)
        self.cnn_model = conv_models.CNNModel(
            grid_planes=grid_planes,
            cnn_sizes=cnn_sizes,
            max_pooling_size=cnn_max_pooling_size,
            batch_norm=cnn_batch_norm,
        )
        test_input = torch.zeros(1, grid_planes, grid_size, grid_size)
        with torch.no_grad():
            test_output = self.cnn_model(test_input)
        self.conv_output_size = list(test_output[0].size())
        self.attention_layer = nn.Linear(embedding_size, cnn_sizes[-1])
        self.flatten = nn_modules.Flatten()

        self.output_model = common_models.OutputFCs(
            input_size=test_output.view(1, -1).size(1),
            embedding_size=embedding_size,
            num_output_fcs=num_output_fcs,
            dropout=dropout
        )

    def forward(self, utts_emb: torch.Tensor, images: torch.Tensor) -> torch.Tensor:
        batch_size = utts_emb.size()[0]

        utts = self.drop(utts_emb)

        images = self.cnn_model(images)
        images = self.drop(images)

        attention = self.attention_layer(utts)
        attention = attention.unsqueeze(-1).unsqueeze(-1).expand(
            batch_size, *self.conv_output_size)
        x = tensor_utils.Hadamard(attention, images)
        x = self.flatten(x)

        x = self.output_model(x)
        return x


class AllPlaneAttentionModel(MultimodalClassifier):
    """
    attention over feature planes in all layers of cnn, not just in the final output
    """
    def __init__(
        self,
        embedding_size: int,
        grid_planes: int,
        grid_size: int,
        cnn_sizes: List[int],
        dropout: float,
        cnn_batch_norm: bool,
        cnn_max_pooling_size: Optional[int],
        num_output_fcs: int,
        **kwargs
    ):
        asserts.assert_values_empty_none_zero(kwargs)
        super().__init__()
        self.embedding_size = embedding_size
        self.grid_size = grid_size
        self.cnn_sizes = cnn_sizes

        self.drop = nn.Dropout(dropout)
        self.conv_with_all_plane_att = common_models.ConvolutionWithAllPlaneAttention(
            grid_planes=grid_planes,
            grid_size=grid_size,
            cnn_sizes=cnn_sizes,
            dropout=dropout,
            batch_norm=cnn_batch_norm,
            max_pooling_size=cnn_max_pooling_size,
        )
        self.attention_layers = nn.ModuleList()
        for i, cnn_size in enumerate(cnn_sizes):
            attention_layer = nn.Linear(embedding_size, cnn_size)
            self.attention_layers.append(attention_layer)
        self.flatten = nn_modules.Flatten()
        self.output_model = common_models.OutputFCs(
            input_size=np.prod(self.conv_with_all_plane_att.size_by_layer_idx[-1]).item(),
            embedding_size=embedding_size,
            num_output_fcs=num_output_fcs,
            dropout=dropout
        )

    def forward(self, utts_emb: torch.Tensor, images: torch.Tensor) -> torch.Tensor:
        utts_emb = self.drop(utts_emb)

        atts = [layer(utts_emb) for layer in self.attention_layers]
        images = self.conv_with_all_plane_att(att_l=atts, grids=images)

        x = self.flatten(images)
        x = self.drop(x)

        x = self.output_model(x)
        return x


class MapUtteranceToConvWeightsModel(MultimodalClassifier):
    def __init__(
        self,
        embedding_size: int,
        grid_planes: int,
        grid_size: int,
        cnn_sizes: List[int],
        cnn_batch_norm: bool,
        cnn_max_pooling_size: Optional[int],
        num_output_fcs: int,
        dropout: float,
        **kwargs
    ):
        asserts.assert_values_empty_none_zero(kwargs)
        super().__init__()
        self.embedding_size = embedding_size
        self.grid_size = grid_size

        self.cnn_sizes = cnn_sizes
        self.cnn_layers = nn.ModuleList()
        # post_cnn_blocks are relu, pooling etc after each conv layer
        self.post_cnn_blocks = nn.ModuleList()

        self.drop = nn.Dropout(dropout)

        last_channels = grid_planes
        for i, cnn_size in enumerate(cnn_sizes):
            self.cnn_layers.append(
                common_models.EmbeddingWeightedConv(
                    weights_embedding_size=embedding_size,
                    in_channels=last_channels, out_channels=cnn_size, kernel_size=3, padding=1))
            post_cnn_block_l: List[nn.Module] = []
            if cnn_batch_norm:
                post_cnn_block_l.append(nn.BatchNorm2d(cnn_size))
            post_cnn_block_l.append(nn.ReLU())
            if cnn_max_pooling_size is not None:
                post_cnn_block_l.append(nn.MaxPool2d(kernel_size=cnn_max_pooling_size))
            self.post_cnn_blocks.append(nn.Sequential(*post_cnn_block_l))
            last_channels = cnn_size

        test_output = torch.zeros(1, grid_planes, grid_size, grid_size)
        test_embedding = torch.zeros(1, embedding_size)
        with torch.no_grad():
            for cnn, post_cnn in zip(self.cnn_layers, self.post_cnn_blocks):
                test_output = cnn(images=test_output, embedding=test_embedding)
                test_output = post_cnn(test_output)

        self.flatten = nn_modules.Flatten()
        self.output_model = common_models.OutputFCs(
            input_size=test_output.view(1, -1).size(1),
            embedding_size=embedding_size,
            num_output_fcs=num_output_fcs,
            dropout=dropout
        )

    def forward(self, utts_emb: torch.Tensor, images: torch.Tensor) -> torch.Tensor:
        batch_size = utts_emb.size()[0]

        for cnn, post_cnn in zip(self.cnn_layers, self.post_cnn_blocks):
            images = cnn(embedding=utts_emb, images=images)
            images = post_cnn(images)

        x = images.view(batch_size, -1)
        x = self.flatten(x)
        x = self.drop(x)

        x = self.output_model(x)
        return x


class LearnedCNNMappingModel(MultimodalClassifier):
    """
    this uses a cnn where the mapping from output planes of one layer to input planes in the next
    is learned. It will be softmaxed, to try to avoid gradient explosions
    the mappings are attention-like, and controlled by the utterances output
    """
    def __init__(
        self,
        embedding_size, cnn_sizes, grid_planes, grid_size, dropout,
        num_output_fcs,
        max_pool_output=False,
    ):
        super().__init__()
        self.embedding_size = embedding_size
        self.grid_size = grid_size
        self.max_pool_output = max_pool_output

        self.drop = nn.Dropout(dropout)
        self.flatten = nn_modules.SpatialToVector()
        print('max_pool_output', max_pool_output)
        print('grid_size', grid_size)
        output_size = 1 if max_pool_output else grid_size
        print('cnn_sizes', cnn_sizes)
        print('output_size', output_size)
        print('num_output_fcs', num_output_fcs)
        print('dropout', dropout)
        self.output_model = common_models.OutputFCs(
            input_size=cnn_sizes[-1] * output_size * output_size,
            embedding_size=embedding_size,
            num_output_fcs=num_output_fcs,
            dropout=dropout
        )
        self.max_pool = nn.MaxPool2d(grid_size)
        self.conv_blocks = nn.ModuleList()
        last_channels = grid_planes
        for i, cnn_size in enumerate(cnn_sizes):
            block = []
            if i != 0 and i != len(cnn_sizes) - 1:
                block.append(nn.Dropout(dropout))
                block.append(nn.ReLU())
            block.append(nn.Conv2d(in_channels=last_channels, out_channels=cnn_size, kernel_size=3, padding=1))
            last_channels = cnn_size
            self.conv_blocks.append(nn.Sequential(*block))

        self.att_blocks = nn.ModuleList()
        """
        this attention is going to run between each conv layer, and before/after
        the first/last conv layer, too
        """

        for i, cnn_size in enumerate([grid_planes] + cnn_sizes):
            attentional_planar_remapping = common_models.AttentionalPlanarRemapping(
                embedding_size=embedding_size,
                num_channels=cnn_size
            )
            self.att_blocks.append(attentional_planar_remapping)

    def forward(self, utts_emb: torch.Tensor, images: torch.Tensor) -> torch.Tensor:
        for i, conv_block in enumerate(self.conv_blocks):
            images = self.att_blocks[i](images, utts_emb)
            images = conv_block(images)
        images = self.att_blocks[-1](images, utts_emb)

        if self.max_pool_output:
            images = self.max_pool(images)
        x = self.flatten(images)
        x = self.drop(x)

        x = self.output_model(x)
        return x


def build_multimodal_classifier(params, ds_meta, pre_conv: pre_conv_lib.PreConv) -> MultimodalClassifier:
    p = params
    MultimodalClassifierClass = globals()[f'{p.multimodal_classifier}Model']
    multimodal_classifier_params = {
        'embedding_size': p.embedding_size,
        'grid_planes': pre_conv.get_output_planes(ds_meta.grid_planes),
        'grid_size': pre_conv.get_output_size(ds_meta.grid_size),
        'dropout': p.dropout,
        'cnn_sizes': p.cnn_sizes,
        'cnn_max_pooling_size': p.cnn_max_pooling_size,
        'cnn_batch_norm': p.cnn_batch_norm
    }
    if p.multimodal_classifier in ['LearnedCNNMapping']:
        del multimodal_classifier_params['cnn_max_pooling_size']
        del multimodal_classifier_params['cnn_batch_norm']
    if MultimodalClassifierClass not in [CosineModel]:
        multimodal_classifier_params['num_output_fcs'] = p.num_output_fcs
    multimodal_classifier: MultimodalClassifier = MultimodalClassifierClass(**multimodal_classifier_params)
    return multimodal_classifier
