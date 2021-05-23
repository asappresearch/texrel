import math
import torch
from torch import nn


class PositionEncoding(nn.Module):
    def __init__(self, max_len, embedding_size, dropout):
        super().__init__()

        self.max_len = max_len
        self.embedding_size = embedding_size
        self.drop = nn.Dropout(dropout)

        pe = torch.zeros(max_len, 1, embedding_size)
        for t in range(max_len):
            for i in range(embedding_size // 2):
                pe[t, 0, i * 2] = math.sin(t / math.pow(10000, 2 * i / embedding_size))
                pe[t, 0, i * 2 + 1] = math.cos(t / math.pow(10000, 2 * i / embedding_size))
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        x = self.drop(x)
        return x
