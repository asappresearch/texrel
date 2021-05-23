import time

import matplotlib.pyplot as plt
import numpy as np
import torch

from ulfs.tensor_utils import Hadamard
from texrel.texturizer import Texturizer


def test_single_images():
    batch_size = 3
    grid_size = 7
    num_objects = 30
    num_textures = 9
    num_colors = 9
    texture_size = 2

    texturizer = Texturizer(
        num_textures=num_textures,
        num_colors=num_colors,
        texture_size=texture_size,
        background_noise=0,
        background_mean=0,
        background_mean_std=0,
        seed=None
    )

    N = batch_size

    mask = torch.zeros(N, grid_size, grid_size, dtype=torch.int64)
    for n in range(N):
        mask_idxes = torch.from_numpy(np.random.choice(grid_size * grid_size, num_objects, replace=False))
        mask.view(N, grid_size * grid_size)[n, mask_idxes] = 1

    idxes = torch.from_numpy(np.random.choice(num_textures * num_colors, (N, grid_size, grid_size), replace=True))
    idxes = Hadamard(idxes, mask)

    t_idxes = idxes // num_colors + 1
    c_idxes = idxes % num_colors + 1

    t_idxes = Hadamard(t_idxes, mask)
    c_idxes = Hadamard(c_idxes, mask)

    grids = texturizer.forward(texture_idxes=t_idxes, color_idxes=c_idxes, savefig=False)

    for n in range(batch_size):
        plt.imshow(grids[n].transpose(-3, -2).transpose(-2, -1).detach().numpy())
        plt.savefig(f'/tmp/grids{n}.png')


def test_sequence():
    batch_size = 3
    grid_size = 7
    num_objects = 30
    num_textures = 9
    num_colors = 9
    texture_size = 2

    texturizer = Texturizer(
        num_textures=num_textures,
        num_colors=num_colors,
        texture_size=texture_size,
        background_noise=0,
        background_mean=0,
        background_mean_std=0,
        seed=None
    )

    M = 5
    N = batch_size

    start_time = time.time()
    mask = torch.zeros(M, N, grid_size, grid_size, dtype=torch.int64)
    for m in range(M):
        for n in range(N):
            mask_idxes = torch.from_numpy(np.random.choice(grid_size * grid_size, num_objects, replace=False))
            mask.view(M, N, grid_size * grid_size)[m, n, mask_idxes] = 1

    idxes = torch.from_numpy(np.random.choice(num_textures * num_colors, (M, N, grid_size, grid_size), replace=True))
    idxes = Hadamard(idxes, mask)

    t_idxes = idxes // num_colors + 1
    c_idxes = idxes % num_colors + 1

    t_idxes = Hadamard(t_idxes, mask)
    c_idxes = Hadamard(c_idxes, mask)
    print('generation time', time.time() - start_time)

    start_time = time.time()
    grids = texturizer.forward(texture_idxes=t_idxes, color_idxes=c_idxes, savefig=False)
    print('texturizer time', time.time() - start_time)

    for m in range(3):
        for n in range(2):
            plt.imshow(grids[m][n].transpose(-3, -2).transpose(-2, -1).detach().numpy())
            plt.savefig(f'/tmp/gridsseq_M{m}_N{n}.png')
