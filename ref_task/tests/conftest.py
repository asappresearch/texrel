import argparse
import pytest


@pytest.fixture
def params() -> argparse.Namespace:
    ds_texture_size = 3
    embedding_size = 7
    dropout = 0
    preconv_dropout = 0.2

    params = argparse.Namespace()
    params.hypothesis_sampler = 'HypothesisSamplerGumbel'
    params.hypothesis_tau = 1.2
    params.hypothesis_noise = 0.0
    params.hypothesis_ent_reg = 0.01
    params.hypothesis_gumbel_hard = True
    params.hypothesis_apply_log = True

    params.ds_texture_size = ds_texture_size

    params.preconv_relu = False
    params.preconv_stride = 4
    params.preconv_model = 'StridedConv'
    params.preconv_dropout = preconv_dropout
    params.preconv_embedding_size = embedding_size
    params.multimodal_classifier = 'LearnedCNNMapping'
    params.embedding_size = embedding_size
    params.dropout = dropout

    params.cnn_sizes = [4, 4]
    params.cnn_max_pooling_size = None
    params.cnn_batch_norm = False

    params.num_output_fcs = 1
    params.linguistic_encoder = 'RNN'
    params.linguistic_encoder_num_layers = 1
    params.linguistic_encoder_rnn_type = 'GRU'

    params.image_seq_embedder = 'RCNN'
    params.sender_negex = True
    params.sender_num_rnn_layers = 1
    params.sender_decoder = 'RNNDecoder'

    params.batch_size = 8
    params.sub_batch_size = 4
    params.phase0_its = 3
    params.phase1_its = 3
    params.hyp_hinge_reg = 0
    params.hyp_l1_reg = 0
    params.hyp_self_norm_l1_reg = 0
    params.hyp_self_norm_l2_reg = 0
    params.hyp_first_symbol_margin_reg = 0
    params.enable_predictor = False
    params.backpropm_randomize_cols = False
    params.backpropm_accumulate_cols = 1

    params.hyp_opt = 'Adam'
    params.hyp_checker_opt = 'Adam'
    params.hyp_inner_opt = 'Adam'
    params.hyp_outer_opt = 'Adam'
    params.meta_inner_opt = 'Adam'
    params.meta_outer_opt = 'Adam'
    params.meta_inner_steps = 2
    params.meta_accumulate_over = 1

    params.meta_lr = 0.001
    params.lr = 0.001

    return params
