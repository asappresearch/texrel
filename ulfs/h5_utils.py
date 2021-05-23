import h5py
import numpy as np
import torch


dt_string = h5py.special_dtype(vlen=str)


def get_value(h5_file, key):
    return h5_file[key][0]


def store_value(h5_file, key, value):
    if isinstance(value, str):
        # dt_string = h5py.special_dtype(vlen=str)
        value_h5 = h5_file.create_dataset(key, (1,), dtype=dt_string)
    elif isinstance(value, int):
        value_h5 = h5_file.create_dataset(key, (1,), dtype=np.int64)
    elif isinstance(value, float):
        value_h5 = h5_file.create_dataset(key, (1,), dtype=np.float32)
    else:
        raise Exception('unhandled value type ' + str(type(value)) + ' for ' + key)
    value_h5[0] = value


def store_tensor(h5_file, key, value):
    size = list(value.size())
    if value.dtype == torch.int8:
        value = value.byte()
    torch_dtype = value.dtype
    print(value.dtype)
    np_dtype = {
        torch.int64: np.int64,
        torch.uint8: np.uint8,
        torch.float32: np.float32,
    }[torch_dtype]
    h5_ds = h5_file.create_dataset(key, size, dtype=np_dtype)
    print('writing', key, '...')
    h5_ds[:] = value
    print(' ... done')


class H5Wrapper(object):
    def __init__(self, h5_f):
        self.h5_f = h5_f

    def store_value(self, key, value):
        store_value(self.h5_f, key, value)

    def store_tensor(self, key, value):
        store_tensor(self.h5_f, key, value)

    def get_value(self, key):
        return get_value(self.h5_f, key)
