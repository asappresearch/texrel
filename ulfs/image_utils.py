import torch
import numpy as np
import PIL


def save_image(filepath: str, image: torch.Tensor) -> None:
    """
    assumes image is [C][H][W], floats
    """
    p_image = PIL.Image.fromarray(
        (image * 255).int().transpose(-3, -2).transpose(-2, -1)
        .detach().numpy().astype(np.uint8))
    p_image.save(filepath)


def upsample_image(up_sample: int, tgt: torch.Tensor) -> torch.Tensor:
    """
    up_sample is integer saying how much larger to make tgt
    tgt is [planes][grid size][grid size]
    """
    grid_planes = tgt.size(0)
    tgt = tgt.unsqueeze(3).unsqueeze(2)
    tgt = tgt.expand(
        grid_planes, tgt.size(1), up_sample,
        tgt.size(3), up_sample)
    res = tgt.contiguous().view(grid_planes, tgt.size(1) * up_sample, -1)
    return res
