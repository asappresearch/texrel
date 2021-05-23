"""
Things that are not the final thing that we pass to --model-type, but used potentially
by one or more of such models
"""
from typing import Sequence, List, Dict, Optional

import torch
from torch import nn
import torch.nn.functional as F

from ulfs import tensor_utils
# from ulfs.stochastic_trajectory import StochasticTrajectory


# class StochasticRNNDecoderFixedLength(nn.Module):
#     """
#     this is kind of redundant to the decoders in decoders_reinforce.py
#     might remove this...
#     """
#     def __init__(
#             self, embedding_size, out_seq_len, vocab_size):
#         super().__init__()
#         # self.input_size = input_size
#         self.embedding_size = embedding_size
#         self.out_seq_len = out_seq_len
#         self.vocab_size = vocab_size

#         # d2e "discrete to embed"
#         self.d2e = nn.Embedding(vocab_size, embedding_size)
#         self.rnn = nn.GRUCell(embedding_size, embedding_size)
#         self.e2d = nn.Linear(embedding_size, vocab_size)

#     def forward(self, x, global_idxes):
#         """
#         x is [N][E]  where [N] is batch size, and [E] is input size
#         """
#         batch_size = x.size(0)
#         device = x.device

#         state = x
#         global_idxes = global_idxes.clone()

#         last_token = torch.zeros(batch_size, dtype=torch.int64, device=device)
#         utterance = torch.zeros(self.out_seq_len, batch_size, dtype=torch.int64, device=device)

#         stochastic_trajectory = None
#         if self.training:
#             stochastic_trajectory = StochasticTrajectory()
#         for t in range(self.out_seq_len):
#             emb = self.d2e(last_token)
#             state = self.rnn(emb, state)
#             token_logits = self.e2d(state)
#             token_probs = F.softmax(token_logits, dim=-1)

#             if self.training:
#                 s = rl_common.draw_categorical_sample(
#                     action_probs=token_probs, batch_idxes=global_idxes)
#                 stochastic_trajectory.append_stochastic_sample(s=s)
#                 token = s.actions.view(-1)
#             else:
#                 _, token = token_probs.max(-1)
#             utterance[t] = token
#             last_token = token

#         return stochastic_trajectory, utterance


class OutputFCs(nn.Module):
    """
    multiple FC layers, one on top of another
    """
    def __init__(self, input_size: int, embedding_size: int, num_output_fcs: int, dropout: float) -> None:
        super().__init__()
        self.embedding_size = embedding_size

        self.layers = nn.ModuleList()
        last_size = input_size
        for i in range(num_output_fcs):
            if i > 0:
                self.layers.append(nn.Dropout(dropout))
                self.layers.append(nn.Tanh())

            if i != num_output_fcs - 1:
                self.layers.append(nn.Linear(last_size, embedding_size))
                last_size = embedding_size
            else:
                self.layers.append(nn.Linear(last_size, 2))

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return super().__call__(x=x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for i, l in enumerate(self.layers):
            x = l(x)
        return x


class ConvolutionWithAllPlaneAttention(nn.Module):
    """
    input on forward:
    - `att_l`: attention over planes, for each conv layer (a list of tensors)
    - `grids`: raw input to conv layer stack

    layers are separated by relu and dropout, including at output, but
    not including at input

    ===> assumes input is [N][C][H][W]  <===
    """
    def __init__(
        self,
        grid_planes: int,
        grid_size: int,
        cnn_sizes: List[int],
        dropout: float,
        batch_norm: bool,
        max_pooling_size: Optional[int],
    ):
        super().__init__()
        self.grid_planes = grid_planes
        self.grid_size = grid_size
        self.cnn_sizes = cnn_sizes

        self.layers = nn.ModuleList()
        last_channels = grid_planes
        self.conv_idx_by_block: Dict[nn.Module, int] = {}
        self.size_by_layer_idx: List[List[int]] = []
        for i, cnn_size in enumerate(cnn_sizes):
            conv_block_l: List[nn.Module] = []
            conv_block_l.append(nn.Conv2d(
                in_channels=last_channels, out_channels=cnn_size, kernel_size=3, padding=1))
            if batch_norm:
                conv_block_l.append(nn.BatchNorm2d(cnn_size))
            conv_block_l.append(nn.ReLU())
            if max_pooling_size is not None:
                conv_block_l.append(nn.MaxPool2d(kernel_size=max_pooling_size))
            conv_block = nn.Sequential(*conv_block_l)
            self.conv_idx_by_block[conv_block] = len(self.conv_idx_by_block)
            self.layers.append(conv_block)
            last_channels = cnn_size
        test_input = torch.zeros(1, grid_planes, grid_size, grid_size)
        with torch.no_grad():
            for block in self.layers:
                test_output = block(test_input)
                self.size_by_layer_idx.append(list(test_output[0].size()))
                test_input = test_output

    def forward(self, att_l: Sequence[torch.Tensor], grids: torch.Tensor):
        """
        grids: [N][C][H][W]
        att_l: I think each tensor is [N][C]
        """
        batch_size = grids.size()[0]
        layer: nn.Module
        for block in self.layers:
            grids = block(grids)
            conv_idx = self.conv_idx_by_block.get(block, None)
            if conv_idx is not None:
                att = att_l[conv_idx]
                _this_grid_size = self.size_by_layer_idx[conv_idx][-1]
                att = att.unsqueeze(-1).unsqueeze(-1).expand(
                    batch_size, self.cnn_sizes[conv_idx], _this_grid_size, _this_grid_size)
                grids = tensor_utils.Hadamard(att, grids)
        return grids


class EmbeddingWeightedConv(nn.Module):
    """
    weights of conv layer come from one input, adn this conv acts
    on the other input

    This is a single convolutional layer (cf stacked multi-layer)
    """
    def __init__(
        self,
        weights_embedding_size: int,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        padding: int = 1
    ):
        super().__init__()
        self.kernel_size = kernel_size
        self.padding = padding
        self.in_channels = in_channels
        self.out_channels = out_channels

        # other kernel sizes wont work currently, because the out_size will be incorrect
        assert self.kernel_size == 3 and self.padding == 1

        self.fc_c1_w = nn.Linear(
            weights_embedding_size, in_channels * out_channels * kernel_size * kernel_size)
        self.fc_c1_b = nn.Linear(
            weights_embedding_size, out_channels)

    def forward(self, embedding: torch.Tensor, images: torch.Tensor):
        """
        embedding: [N][E]
        images: [N][C][H][W]
        """
        x1 = embedding
        x2 = images
        batch_size = x1.size()[0]
        assert len(x2.size()) == 4
        in_size = x2.size()[2]
        out_size = in_size  # assumes zero padded

        c1_w = self.fc_c1_w(x1)
        c1_b = self.fc_c1_b(x1)
        c1_b = c1_b.unsqueeze(-1).unsqueeze(-1)

        res = torch.zeros((batch_size, self.out_channels, out_size, out_size), device=embedding.device)
        for n in range(batch_size):
            out_unfold = F.unfold(x2[n: n+1], kernel_size=self.kernel_size, padding=self.padding)  # type: ignore
            _c1_w = c1_w[n]
            _c1_w = _c1_w.view(self.out_channels, -1)
            _res = _c1_w @ out_unfold
            _res = _res.view(1, self.out_channels, out_size, out_size)
            _res += c1_b[n]
            res[n:n+1] = _res
        return res


class AttentionalPlanarRemappingOld(nn.Module):
    """
    So, given some embedding_size'd embedding, going to remap
    the incoming planes, and output the remapped spatial embedding
    """
    def __init__(self):
        super().__init__()

    def forward(self, images, atts):
        """
        images are expected to be [N][C][H][W]
        atts should be [N][C][C]
        """
        grids = images

        N, C, H, W = grids.size()
        atts = atts.view(N, C, C)
        atts = F.softmax(atts, dim=-1)

        atts = atts.unsqueeze(-1).unsqueeze(-1).expand(N, C, C, H, W)
        grids = grids.view(N, 1, C, H, W).expand(N, C, C, H, W)
        grids_out = tensor_utils.Hadamard(atts, grids)
        grids_out = grids_out.sum(dim=2)
        return grids_out


class AttentionalPlanarRemapping(nn.Module):
    """
    So, given some embedding_size'd embedding, going to remap
    the incoming planes, and output the remapped spatial embedding
    """
    def __init__(self, embedding_size, num_channels):
        super().__init__()
        block = []
        # this attention is going to run at the *end* of each conv block,
        # we need an attention weight for each pair of output/input
        # planes
        h = nn.Linear(embedding_size, num_channels * num_channels)

        block.append(h)
        block.append(nn.Softmax(dim=-1))
        self.att_block = (nn.Sequential(*block))

    def forward(self, images, atts):
        """
        images are expected to be [N][C][H][W]
        # atts should be [N][C][C]
        atts should be [N][E]
        """
        grids = images
        N, C, H, W = grids.size()

        atts = self.att_block(atts)
        atts = atts.view(N, C, C)
        atts = F.softmax(atts, dim=-1)

        atts = atts.unsqueeze(-1).unsqueeze(-1).expand(N, C, C, H, W)
        grids = grids.view(N, 1, C, H, W).expand(N, C, C, H, W)
        grids_out = tensor_utils.Hadamard(atts, grids)
        grids_out = grids_out.sum(dim=2)
        return grids_out
