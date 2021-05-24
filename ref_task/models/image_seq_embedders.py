"""
takes a sequence of input images, and embeds them

doesnt itself run decoding

ie output is a simple embedding_size'd embedding, not a sequence,
not tokens, etc

can be used by both differentiable and reinforce decoders, etc

these classes mostly don't themselves know anything about labels,
unless label_aware is set to True.
Where label_aware is set to False, calling classes might add labels into the
input image stack as additional feature planes
"""
import torch
from torch import nn
from typing import List, Optional
import torch.nn.functional as F

from ulfs import nn_modules, tensor_utils
from ref_task.models import conv_models
from ref_task.models.image_seq_embedder_base import ImageSeqEmbedder


class RNNOverCNN(ImageSeqEmbedder):
    """
    pass each image/grid through a CNN, into an RNN
    pass output of RNN through MDP heads, output to relation
    """
    label_aware = False

    def __init__(
                self, embedding_size: int, dropout: float, cnn_sizes: List[int],
                grid_planes: int, grid_size: int, cnn_max_pooling_size: Optional[int],
                cnn_batch_norm: bool
            ):
        super().__init__()
        self.embedding_size = embedding_size
        self.dropout = dropout
        self.cnn_sizes = cnn_sizes
        self.grid_planes = grid_planes
        self.grid_size = grid_size

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
        self.drop = nn.Dropout(dropout)
        self.rnn_model = nn.LSTMCell(test_output.size(1), embedding_size)

    def forward(self, images):
        """
        grids should be [T][N][C][H][W]  ([T] => timesteps (seqlen))
        """
        grids = images
        seq_len = grids.size()[0]
        batch_size = grids.size()[1]
        torch_constr = torch.cuda if grids.is_cuda else torch

        rnn_state = torch_constr.FloatTensor(batch_size, self.embedding_size).zero_()
        rnn_cell = torch_constr.FloatTensor(batch_size, self.embedding_size).zero_()
        for t in range(seq_len):
            x = grids[t]
            x = self.cnn_model(x)
            x = self.flatten(x)
            x = self.drop(x)
            rnn_state, rnn_cell = self.rnn_model(x, (rnn_state, rnn_cell))
            rnn_state = self.drop(rnn_state)
        rnn_out = rnn_state
        return rnn_out


class RCNN(ImageSeqEmbedder):
    """
    uses an rnn built from cnn layers :)
    """

    label_aware = False

    def __init__(
                self, embedding_size: int, dropout: float, cnn_sizes: List[int],
                grid_planes: int, grid_size: int,
                num_rnn_layers: int, cnn_max_pooling_size: Optional[int], cnn_batch_norm: bool
            ):
        super().__init__()
        self.embedding_size = embedding_size
        self.dropout = dropout
        self.cnn_sizes = cnn_sizes
        self.grid_planes = grid_planes
        self.grid_size = grid_size
        self.num_rnn_layers = num_rnn_layers

        self.hidden_planes = cnn_sizes[-1]

        def cnn_constr(input_planes, _):
            cnn = conv_models.CNNModel(
                grid_planes=input_planes,
                cnn_sizes=cnn_sizes,
                batch_norm=cnn_batch_norm,
                max_pooling_size=cnn_max_pooling_size)
            return cnn

        input_planes = grid_planes
        self.rcnn = nn_modules.RCNN(
            cnn_constr=cnn_constr,
            input_planes=input_planes,
            hidden_planes=self.hidden_planes,
            dropout=dropout,
            num_layers=num_rnn_layers,
            grid_size=grid_size
        )
        self.drop = nn.Dropout(dropout)
        self.flatten = nn_modules.Flatten()
        # self.relation_head = nn.Linear(self.hidden_planes * grid_size * grid_size, vocab_size * out_seq_len)

        test_input = torch.zeros(1, grid_planes, grid_size, grid_size)
        with torch.no_grad():
            test_cnn = cnn_constr(input_planes=input_planes, _=None)
            test_output = test_cnn(test_input).view(1, -1)

        self.flat_spatial_size = test_output.size(1)
        self.h1 = nn.Linear(self.flat_spatial_size, embedding_size)

    def forward(self, images):
        """
        grids should be [T][N][C][H][W]  ([T] => timesteps (seqlen))
        """
        grids = images

        out, (_, _) = self.rcnn(grids)

        x = self.flatten(out[-1])
        x = self.drop(x)
        x = self.h1(x)
        return x


class StackedInputs(ImageSeqEmbedder):
    """
    Simply stack/concatenate the input examples together, feed into a standard cnn.
    In the worst case, it's an important baseline; in the best case it works ok
    (as long as we have fixed input sizes, which ... we do :/ )
    """

    label_aware = False

    def __init__(
        self,
        embedding_size: int,
        dropout: float,
        cnn_sizes: List[int],
        grid_planes: int,
        grid_size: int,
        num_rnn_layers: int,
        cnn_max_pooling_size: Optional[int],
        cnn_batch_norm: bool,
        input_examples: int,   # this model needs to know this in advance...
    ):
        super().__init__()
        self.embedding_size = embedding_size  # not used currently (but might, if we have additional fc output layers)
        self.grid_planes = grid_planes
        self.grid_size = grid_size

        self.cnn = conv_models.CNNModel(
            grid_planes=grid_planes * input_examples,
            cnn_sizes=cnn_sizes,
            max_pooling_size=cnn_max_pooling_size,
            batch_norm=cnn_batch_norm)

        test_input = torch.zeros(1, grid_planes * input_examples, grid_size, grid_size)
        with torch.no_grad():
            test_output = self.cnn(test_input).view(1, -1)

        self.drop = nn.Dropout(dropout)
        self.flatten = nn_modules.Flatten()
        self.h1 = nn.Linear(test_output.size(1), embedding_size)

    def forward(self, images):
        """
        grids input assumed to be [T][N][C][H][W] (T is seq_len)
        """
        batch_size = images.size()[1]
        seq_len = images.size()[0]

        """
        reshape grids from [T][N][C][H][W]
        to                 [N][T * C][H][W]
        """
        images = images.transpose(0, 1).contiguous()
        images = images.view(batch_size, seq_len * self.grid_planes, self.grid_size, self.grid_size)

        images = self.cnn(images)
        x = self.flatten(images)
        x = self.drop(x)
        x = self.h1(x)
        return x


class PoolingCNN(ImageSeqEmbedder):
    """
    run each input example through CNN, then pool
    """

    label_aware = False

    def __init__(
        self,
        embedding_size: int,
        dropout: float,
        cnn_sizes: List[int],
        cnn_max_pooling_size: Optional[int],
        cnn_batch_norm: bool,
        grid_planes: int,
        grid_size: int,
        pooling_fn,
    ):
        super().__init__()
        self.embedding_size = embedding_size
        self.grid_planes = grid_planes
        self.grid_size = grid_size
        self.cnn_sizes = cnn_sizes
        self.pooling_fn = pooling_fn

        self.cnn = None
        self.cnn = conv_models.CNNModel(
            grid_planes=grid_planes,
            cnn_sizes=cnn_sizes,
            max_pooling_size=cnn_max_pooling_size,
            batch_norm=cnn_batch_norm)
        with torch.no_grad():
            sample_image = torch.zeros(1, grid_planes, grid_size, grid_size)
        sample_out = self.cnn(sample_image)
        _, output_C, output_H, output_W = sample_out.size()
        self.drop = nn.Dropout(dropout)
        self.flatten = nn_modules.Flatten()
        self.h1 = nn.Linear(output_C * output_H * output_W, embedding_size)

    def forward(self, images):
        """
        images input assumed to be [T][N][C][H][W] (T is seq_len)
        """
        seq_len, batch_size, channels, height, width = images.size()
        if self.cnn is not None:
            merger = tensor_utils.DimMerger()
            images_flat = merger.merge(images, dim1=0, dim2=1)
            images_flat = self.cnn(images_flat)
            _, C, H, W = images_flat.size()
            images = merger.resplit(images_flat, dim=0)

        # images is now [T][N][C][H][W]
        images = images.transpose(0, 1).transpose(1, 2).transpose(2, 3).transpose(3, 4).view(batch_size, -1, seq_len)
        # images is now [N][C * H * W][T]
        # pool over [T]
        images = self.pooling_fn(images, kernel_size=seq_len)
        # images is now [N][C * H * W][1]
        images = images.squeeze(-1).view(batch_size, C, H, W)
        # images is now [N][C][H][W]

        x = self.flatten(images)
        # x is now [N][C * H * W]
        x = self.drop(x)
        x = self.h1(x)
        # x is now [N][E]
        return x


class MaxPoolingCNN(PoolingCNN):
    """
    run each input example through CNN, then max pool
    """

    label_aware = False

    def __init__(
        self,
        embedding_size: int,
        dropout: float,
        cnn_sizes: List[int],
        grid_planes: int,
        grid_size: int,
        cnn_max_pooling_size: Optional[int], cnn_batch_norm: bool
    ):
        super().__init__(
            embedding_size=embedding_size,
            dropout=dropout,
            cnn_sizes=cnn_sizes,
            grid_planes=grid_planes,
            grid_size=grid_size,
            cnn_max_pooling_size=cnn_max_pooling_size,
            cnn_batch_norm=cnn_batch_norm,
            pooling_fn=F.max_pool1d)


class AveragePoolingCNN(PoolingCNN):
    """
    run each input example through CNN, then average pool
    """

    label_aware = False

    def __init__(
        self,
        embedding_size: int,
        dropout: float,
        cnn_sizes: List[int],
        grid_planes: int,
        grid_size: int,
        cnn_max_pooling_size: Optional[int], cnn_batch_norm: bool
    ):
        super().__init__(
            embedding_size=embedding_size,
            dropout=dropout,
            cnn_sizes=cnn_sizes,
            grid_planes=grid_planes,
            grid_size=grid_size,
            cnn_max_pooling_size=cnn_max_pooling_size,
            cnn_batch_norm=cnn_batch_norm,
            pooling_fn=F.avg_pool1d)


def mask_images(images: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """
    images is [M][N][...stuff...]
    mask is [M][N]

    the returned tensor should be squashed along M dimension to only
    return the things where mask is 1
    the number of things where mask is 1 should be the same for
    each value of 1..N
    """
    images_size = list(images.size())
    mask_sum = mask.sum(dim=0)
    num_pos: int = mask_sum[0].item()  # type: ignore
    assert (mask_sum == num_pos).all()
    images = images.transpose(0, 1)[mask.transpose(0, 1)]
    images = images.view(images_size[1], num_pos, *images_size[2:]).transpose(0, 1).contiguous()
    return images


class PrototypicalSender(nn.Module):
    """
    This will use prototypical networks, as in L3 and LSL L3, to create an embedding
    for the positive example class

    we only consider a single positive-example class
    """
    label_aware = True

    def __init__(
        self,
        embedding_size: int,
        dropout: float,
        cnn_sizes: List[int],
        grid_planes: int,
        grid_size: int,
        cnn_max_pooling_size: Optional[int],
        cnn_batch_norm: bool
    ):
        super().__init__()
        self.embedding_size = embedding_size
        self.dropout = dropout
        self.cnn_sizes = cnn_sizes
        self.grid_planes = grid_planes
        self.grid_size = grid_size

        self.average_pooler = AveragePoolingCNN(
            embedding_size=embedding_size,
            dropout=dropout,
            cnn_sizes=cnn_sizes,
            grid_planes=grid_planes,
            grid_size=grid_size,
            cnn_max_pooling_size=cnn_max_pooling_size,
            cnn_batch_norm=cnn_batch_norm)

    def __call__(self, images, labels) -> torch.Tensor:
        return super().__call__(images=images, labels=labels)

    def forward(self, images, labels) -> torch.Tensor:
        # we assume positive labels are 1
        # assume each row has same number of positive examples
        images = mask_images(images=images, mask=labels == 1)
        images_emb = self.average_pooler(images)
        return images_emb


def build_image_seq_embedder(params, ds_meta, pre_conv, utt_len: int, vocab_size: int) -> ImageSeqEmbedder:
    p = params
    # if p.image_seq_embedder == 'RelationsTransformer':
    #     ImageSeqEmbedder = RelationsTransformer
    #     assert p.sender_decoder == 'IdentityDecoder'
    # else:
    ImageSeqEmbedder = globals()[p.image_seq_embedder]
    grid_planes = pre_conv.get_output_planes(ds_meta.grid_planes)
    if p.sender_negex and not ImageSeqEmbedder.label_aware:
        grid_planes += 1
    model_params = {
        'embedding_size': p.embedding_size,
        'grid_planes': grid_planes,
        'grid_size': pre_conv.get_output_size(ds_meta.grid_size),
        'dropout': p.dropout,
        'cnn_sizes': p.cnn_sizes,
        'cnn_max_pooling_size': p.cnn_max_pooling_size,
        'cnn_batch_norm': p.cnn_batch_norm,
    }
    # if ImageSeqEmbedder == RelationsTransformer:
    #     model_params['num_heads'] = p.sender_num_heads
    #     model_params['num_timesteps'] = p.sender_num_timesteps
    #     # no decoder for RelationsTransformer (just use Identity), so
    #     # we have to give the desired seq_len and vocab size directly
    #     # to the RelationsTransformer, here
    #     model_params['out_seq_len'] = utt_len
    #     model_params['vocab_size'] = vocab_size
    #     # del model_params['cnn_sizes']
    #     # del model_params['cnn_max_pooling_size']
    #     # del model_params['cnn_batch_norm']
    #     del model_params['dropout']
    if ImageSeqEmbedder in [RCNN, StackedInputs]:
        model_params['num_rnn_layers'] = p.sender_num_rnn_layers
        if ImageSeqEmbedder == StackedInputs:
            model_params['input_examples'] = ds_meta.M_train
    try:
        image_seq_embedder = ImageSeqEmbedder(**model_params)
    except Exception as e:
        print('ImageSeqEmbedder', ImageSeqEmbedder)
        print('model_params', model_params)
        raise e
    return image_seq_embedder
