import math

from ulfs import gru_utils


FACTOR_RELU = 1.43


def susillo_init_linear(linear, factor=1):
    input_size = linear.weight.size(1)
    linear.bias.data[:] = 0
    rng = math.sqrt(3) / math.sqrt(input_size) * factor
    linear.weight.data.uniform_(-rng, rng)


def susillo_init_embedding(embedding, factor=1):
    input_size = embedding.weight.size(0)
    rng = math.sqrt(3) / math.sqrt(input_size) * factor
    embedding.weight.data.uniform_(-rng, rng)


def susillo_initialize_gru_reset_weight(gru_cell, factor, num_inputs):
    rng = math.sqrt(3) / math.sqrt(num_inputs) * factor
    gru_utils.get_gru_weight_ir(gru_cell).data.uniform_(-rng, rng)
    gru_utils.get_gru_weight_hr(gru_cell).data.uniform_(-rng, rng)


def susillo_initialize_gru_update_weight(gru_cell, factor, num_inputs):
    rng = math.sqrt(3) / math.sqrt(num_inputs) * factor
    gru_utils.get_gru_weight_iz(gru_cell).data.uniform_(-rng, rng)
    gru_utils.get_gru_weight_hz(gru_cell).data.uniform_(-rng, rng)


def susillo_initialize_gru_candidate_weight(gru_cell, factor, num_inputs):
    rng = math.sqrt(3) / math.sqrt(num_inputs) * factor
    gru_utils.get_gru_weight_in(gru_cell).data.uniform_(-rng, rng)
    gru_utils.get_gru_weight_hn(gru_cell).data.uniform_(-rng, rng)


def constant_initialize_gru_reset_bias(gru_cell, value):
    gru_utils.get_gru_bias_ir(gru_cell).data.fill_(value)
    gru_utils.get_gru_bias_hr(gru_cell).data.fill_(value)


def constant_initialize_gru_update_bias(gru_cell, value):
    gru_utils.get_gru_bias_iz(gru_cell).data.fill_(value)
    gru_utils.get_gru_bias_hz(gru_cell).data.fill_(value)


def constant_initialize_gru_candidate_bias(gru_cell, value):
    gru_utils.get_gru_bias_in(gru_cell).data.fill_(value)
    gru_utils.get_gru_bias_hn(gru_cell).data.fill_(value)


def init_gru_cell(gru_cell):
    hidden_size = gru_cell.bias_hh.data.size(0) // 3
    input_size = gru_cell.weight_ih.data.size(1)
    print('gru input_size', input_size, 'hidden_size', hidden_size)

    susillo_initialize_gru_reset_weight(gru_cell, factor=1, num_inputs=input_size + hidden_size)
    susillo_initialize_gru_update_weight(gru_cell, factor=1, num_inputs=input_size + hidden_size)
    susillo_initialize_gru_candidate_weight(gru_cell, factor=1, num_inputs=(hidden_size * 3) // 2)

    constant_initialize_gru_reset_bias(gru_cell, value=1)
    constant_initialize_gru_update_bias(gru_cell, value=1)
    constant_initialize_gru_candidate_bias(gru_cell, value=0)
