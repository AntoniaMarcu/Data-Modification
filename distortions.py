import math
import random

import numpy as np
import torch
import torchvision
from torchbearer.callbacks.cutout import BatchCutout

from fmix import sample_mask


def mixup(image1: torch.Tensor, image2: torch.Tensor, lam) -> torch.Tensor:
    """ Performs mixup on two images with a mixing factor of lam.

    Args:
        image1: torch.Tensor of shape [n_channels X width X height]
        image2: torch.Tensor of shape [n_channels X width X height]
        lam: float in [0,1]. Mixing factor

    Returns:
        torch.Tensor of shape [n_channels X width X height] representing
        the newly obtained sample
    """
    return image1 * lam + image2 * (1 - lam)


def cutMix_restricted(image1: torch.Tensor, image2: torch.Tensor, lam) -> torch.Tensor:
    """ Performs CutMix where the occluding rectangles are restricted to
    lie within the image boundaries such that the occluding factor is exact.

    image1: torch.Tensor of shape [n_channels X width X height]. Image to be occluded
    image2: torch.Tensor of shape [n_channels X width X height]. Image from which
            occluding patches are sampled
    lam: float in [0,1]. Mixing factor

    Returns:
        torch.Tensor of shape [n_channels X width X height] representing
        the newly obtained sample
    """
    batch1 = image1.reshape(1, *image1.shape)
    batch2 = image2.reshape(1, *image2.shape)

    length = torch.tensor(math.sqrt(lam))
    cutter = BatchCutoutRestricted(1, (length * batch1.size(-1)).round().item(), (length * batch1.size(-2)).round().item())
    mask = cutter(batch1)
    erase_locations = mask == 0

    batch1[erase_locations] = batch2[erase_locations]
    return batch1[0]


def cutMix_unrestricted(image1: torch.Tensor, image2: torch.Tensor, lam) -> torch.Tensor:
    """ Performs CutMix where the occluding rectangles can lie outside
    the image boundaries. As a result, the total occluded area can be
    less than the chosen mixing factor lam.

    image1: torch.Tensor of shape [n_channels X width X height]. Image to be occluded
    image2: torch.Tensor of shape [n_channels X width X height]. Image from which
            occluding patches are sampled
    lam: float in [0,1]. Mixing factor

    Returns:
        torch.Tensor of shape [n_channels X width X height] representing
        the newly obtained sample
    """
    batch1 = image1.reshape(1, image1.shape[0], image1.shape[1], image1.shape[2])
    batch2 = image2.reshape(1, image2.shape[0], image2.shape[1], image2.shape[2])

    length = torch.tensor(math.sqrt(lam))
    cutter = BatchCutout(1, (length * batch1.size(-1)).round().item(), (length * batch1.size(-2)).round().item())
    mask = cutter(batch1)
    erase_locations = mask == 0

    batch1[erase_locations] = batch2[erase_locations]
    return batch1[0]


def cutOut_restricted(image1: torch.Tensor, image2: torch.Tensor, lam) -> torch.Tensor:
    batch = image1.reshape(1, image1.shape[0], image1.shape[1], image1.shape[2])

    length = torch.tensor(math.sqrt(lam))
    cutter = BatchCutoutRestricted(1, (length * batch.size(-1)).round().item(), (length * batch.size(-2)).round().item())
    mask = cutter(batch)
    erase_locations = mask == 0

    batch[erase_locations] = 0

    return batch[0]


def cutOut_unrestricted(image1: torch.Tensor, image2: torch.Tensor, lam) -> torch.Tensor:
    batch = image1.reshape(1, image1.shape[0], image1.shape[1], image1.shape[2])

    length = torch.tensor(math.sqrt(lam))
    cutter = BatchCutout(1, (length * batch.size(-1)).round().item(), (length * batch.size(-2)).round().item())
    mask = cutter(batch)
    erase_locations = mask == 0

    batch[erase_locations] = 0

    return batch[0]


def none(image1: torch.Tensor, image2: torch.Tensor, lam) -> torch.Tensor:
    return image1


class BatchCutoutRestricted(object):
    """
    Randomly mask out one or more patches from a batch of images.
    Code based on torchbearer's CutOut implementation
    (https://torchbearer.readthedocs.io/en/latest/_modules/torchbearer/callbacks/cutout.html#Cutout)

    Args:
        n_holes (int): Number of patches to cut out of each image.
        width (int): The width (in pixels) of each square patch.
        height (int): The height (in pixels) of each square patch.
    """
    def __init__(self, n_holes, width, height):
        self.n_holes = n_holes
        self.width = width
        self.height = height

    def __call__(self, img):
        """
        Args:
            img (Tensor): Tensor image of size (B, C, H, W).
        Returns:
            Tensor: Image with n_holes of dimension length x length cut out of it.
        """
        b = img.size(0)
        c = img.size(1)
        h = img.size(-2)
        w = img.size(-1)

        mask = torch.ones((b, h, w), device=img.device)

        for n in range(self.n_holes):
            y = torch.randint(low=round(self.height/2), high=h - round(self.height/2), size=(b,)).long() if not \
                round(self.height/2) == round(h/2) else  torch.tensor([int(self.height/2)])
            x = torch.randint(low=round(self.width/2), high=w - round(self.width/2), size=(b,)).long() if not \
                round(self.width/2) == round(w/2) else  torch.tensor([int(self.width/2)])

            y1 = (y - self.height // 2).clamp(0, h).type(torch.uint8)
            y2 = (y + self.height // 2).clamp(0, h).type(torch.uint8)
            x1 = (x - self.width // 2).clamp(0, w).type(torch.uint8)
            x2 = (x + self.width // 2).clamp(0, w).type(torch.uint8)

            for batch in range(b):
                mask[batch, y1[batch]: y2[batch], x1[batch]: x2[batch]] = 0

        mask = mask.unsqueeze(1).repeat(1, c, 1, 1)

        return mask


def fmix(img1, img2, lam, decay_power=3, shape=(64, 64), max_soft=0.0, reformulate=False):

    mask = torch.Tensor(sample_mask(lam, decay_power, shape, max_soft, reformulate))

    x1, x2 = img1 * mask, img2 * (1-mask)
    return x1+x2


def fout(img1, img2, lam, decay_power=3, shape=(64, 64), max_soft=0.0, reformulate=False):

    mask = torch.Tensor(sample_mask(lam, decay_power, shape, max_soft, reformulate))

    return img1 * mask


# the patch_shuffle implementation below is based on https://github.com/fmcarlucci/JigenDG
def make_grid(x, grid_size):
    return torchvision.utils.make_grid(x, grid_size, padding=0)


def get_tile_simple(img, n, grid_size):
    w = int(img.shape[1] / grid_size)
    y = int(n / grid_size)
    x = n % grid_size
    tile = img[:, x * w:(x + 1) * w, y * w:(y + 1) * w]
    return tile


def patch_shuffle(img1, img2, grid_size):
    n_grids = grid_size ** 2
    tiles = [None] * n_grids
    for n in range(n_grids):
        tiles[n] = get_tile_simple(img1, n, grid_size)
    permutations = np.arange(0, n_grids)
    random.shuffle(permutations)
    data = [tiles[permutations[t]] for t in range(n_grids)]
    data = torch.stack(data, 0)
    return make_grid(data, grid_size)