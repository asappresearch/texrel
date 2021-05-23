"""
Functions to get to weights and biases inside GRUCell
"""


def get_gru_weight_in(gru_cell):
    gru_size = gru_cell.bias_hh.data.size(0) // 3
    return gru_cell.weight_ih[2 * gru_size:]


def get_gru_weight_iz(gru_cell):
    gru_size = gru_cell.bias_hh.data.size(0) // 3
    return gru_cell.weight_ih[gru_size:2 * gru_size]


def get_gru_weight_ir(gru_cell):
    gru_size = gru_cell.bias_hh.data.size(0) // 3
    return gru_cell.weight_ih[0: gru_size]


def get_gru_bias_in(gru_cell):
    gru_size = gru_cell.bias_hh.data.size(0) // 3
    return gru_cell.bias_ih[2 * gru_size:]


def get_gru_bias_iz(gru_cell):
    gru_size = gru_cell.bias_hh.data.size(0) // 3
    return gru_cell.bias_ih[gru_size:2 * gru_size]


def get_gru_bias_ir(gru_cell):
    gru_size = gru_cell.bias_hh.data.size(0) // 3
    return gru_cell.bias_ih[0: gru_size]


def get_gru_weight_hn(gru_cell):
    gru_size = gru_cell.bias_hh.data.size(0) // 3
    return gru_cell.weight_hh[2 * gru_size:]


def get_gru_weight_hz(gru_cell):
    gru_size = gru_cell.bias_hh.data.size(0) // 3
    return gru_cell.weight_hh[gru_size:2 * gru_size]


def get_gru_weight_hr(gru_cell):
    gru_size = gru_cell.bias_hh.data.size(0) // 3
    return gru_cell.weight_hh[0: gru_size]


def get_gru_bias_hn(gru_cell):
    gru_size = gru_cell.bias_hh.data.size(0) // 3
    return gru_cell.bias_hh[2 * gru_size:]


def get_gru_bias_hz(gru_cell):
    gru_size = gru_cell.bias_hh.data.size(0) // 3
    return gru_cell.bias_hh[gru_size:2 * gru_size]


def get_gru_bias_hr(gru_cell):
    gru_size = gru_cell.bias_hh.data.size(0) // 3
    return gru_cell.bias_hh[0: gru_size]
