import torch

from ref_task.models import image_seq_embedders


def test_mask_images():
    M = 4
    N = 3
    K = 5
    images = torch.rand(M, N, K)
    print('images', images)
    mask = torch.BoolTensor([
        [0, 1, 1, 0],
        [1, 0, 1, 0],
        [0, 1, 1, 0],
    ]).transpose(0, 1)
    res = image_seq_embedders.mask_images(images=images, mask=mask)
    print('res', res)
    print('res.size()', res.size())
    assert list(res.size()) == [2, 3, 5]
    assert res[0, 0, 0] == images[1, 0, 0]
    assert res[1, 0, 0] == images[2, 0, 0]
    assert res[0, 1, 0] == images[0, 1, 0]
    assert res[1, 1, 0] == images[2, 1, 0]
