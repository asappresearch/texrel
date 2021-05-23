import numpy as np
import torch


def create_length_mask(utt_len: int, batch_size: int) -> torch.Tensor:
    length_drop_lengths = torch.from_numpy(np.random.choice(
        utt_len, batch_size, replace=True
    ))
    cumsum = torch.ones(utt_len, batch_size, dtype=torch.int64).cumsum(dim=0) - 1
    length_drop_lengths = length_drop_lengths.unsqueeze(0).expand(utt_len, batch_size)
    mask = cumsum - 1 < length_drop_lengths
    return mask
