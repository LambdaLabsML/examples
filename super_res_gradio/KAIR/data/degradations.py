from typing import Tuple

import numpy as np
import random
import torch
from numpy.typing import NDArray

from basicsr.data.degradations import random_add_gaussian_noise_pt, random_add_poisson_noise_pt
from basicsr.data.transforms import paired_random_crop
from basicsr.utils import DiffJPEG, USMSharp
from basicsr.utils.img_process_util import filter2D
from torch import Tensor
from torch.nn import functional as F


def blur(img: Tensor, kernel: NDArray) -> Tensor:
    return filter2D(img, kernel)


def random_resize(
    img: Tensor,
    resize_prob: float,
    resize_range: Tuple[int, int],
    output_scale: float = 1
) -> Tensor:
    updown_type = random.choices(['up', 'down', 'keep'], resize_prob)[0]
    if updown_type == 'up':
        random_scale = np.random.uniform(1, resize_range[1])
    elif updown_type == 'down':
        random_scale = np.random.uniform(resize_range[0], 1)
    else:
        random_scale = 1
    mode = random.choice(['area', 'bilinear', 'bicubic'])
    out = F.interpolate(img, scale_factor=output_scale * random_scale, mode=mode)
    return out


def add_noise(
    img: Tensor,
    gray_noise_prob: float,
    gaussian_noise_prob: float,
    noise_range: Tuple[float, float],
    poisson_scale_range: Tuple[float, float]
) -> Tensor:
    if np.random.uniform() < gaussian_noise_prob:
        img = random_add_gaussian_noise_pt(
            img, sigma_range=noise_range, clip=True, rounds=False,
            gray_prob=gray_noise_prob)
    else:
        img = random_add_poisson_noise_pt(
            img, scale_range=poisson_scale_range,
            gray_prob=gray_noise_prob, clip=True, rounds=False)
    return img


def jpeg_compression_simulation(
    img: Tensor,
    jpeg_range: Tuple[float, float],
    jpeg_simulator: DiffJPEG
) -> Tensor:
    jpeg_p = img.new_zeros(img.size(0)).uniform_(*jpeg_range)

    # clamp to [0, 1], otherwise JPEGer will result in unpleasant artifacts
    img = torch.clamp(img, 0, 1)
    return jpeg_simulator(img, quality=jpeg_p)


@torch.no_grad()
def apply_real_esrgan_degradations(
    gt: Tensor,
    blur_kernel1: NDArray,
    blur_kernel2: NDArray,
    second_blur_prob: float,
    sinc_kernel: NDArray,
    resize_prob1: float,
    resize_prob2: float,
    resize_range1: Tuple[int, int],
    resize_range2: Tuple[int, int],
    gray_noise_prob1: float,
    gray_noise_prob2: float,
    gaussian_noise_prob1: float,
    gaussian_noise_prob2: float,
    noise_range: Tuple[float, float],
    poisson_scale_range: Tuple[float, float],
    jpeg_compression_range1: Tuple[float, float],
    jpeg_compression_range2: Tuple[float, float],
    jpeg_simulator: DiffJPEG,
    random_crop_gt_size: 512,
    sr_upsample_scale: float,
    usm_sharpener: USMSharp
):
    """
    Accept batch from batchloader, and then add two-order degradations
    to obtain LQ images.

    gt: Tensor of shape (B x C x H x W)
    """
    gt_usm = usm_sharpener(gt)
    # from PIL import Image
    # Image.fromarray((gt_usm[0].permute(1, 2, 0).cpu().numpy() * 255.).astype(np.uint8)).save(
    #         "/home/cll/Desktop/GT_USM_orig.png")
    orig_h, orig_w = gt.size()[2:4]

    # ----------------------- The first degradation process ----------------------- #
    out = blur(gt_usm, blur_kernel1)
    out = random_resize(out, resize_prob1, resize_range1)
    out = add_noise(out, gray_noise_prob1, gaussian_noise_prob1, noise_range, poisson_scale_range)
    out = jpeg_compression_simulation(out, jpeg_compression_range1, jpeg_simulator)

    # ----------------------- The second degradation process ----------------------- #
    if np.random.uniform() < second_blur_prob:
        out = blur(out, blur_kernel2)
    out = random_resize(out, resize_prob2, resize_range2, output_scale=(1/sr_upsample_scale))
    out = add_noise(out, gray_noise_prob2, gaussian_noise_prob2,
                    noise_range, poisson_scale_range)

    # JPEG compression + the final sinc filter
    # We also need to resize images to desired sizes.
    # We group [resize back + sinc filter] together
    # as one operation.
    # We consider two orders:
    #   1. [resize back + sinc filter] + JPEG compression
    #   2. JPEG compression + [resize back + sinc filter]
    # Empirically, we find other combinations (sinc + JPEG + Resize)
    # will introduce twisted lines.
    if np.random.uniform() < 0.5:
        # resize back + the final sinc filter
        mode = random.choice(['area', 'bilinear', 'bicubic'])
        out = F.interpolate(out, size=(orig_h // sr_upsample_scale,
                                       orig_w // sr_upsample_scale), mode=mode)
        out = blur(out, sinc_kernel)
        out = jpeg_compression_simulation(out, jpeg_compression_range2, jpeg_simulator)
    else:
        out = jpeg_compression_simulation(out, jpeg_compression_range2, jpeg_simulator)
        mode = random.choice(['area', 'bilinear', 'bicubic'])
        out = F.interpolate(out, size=(orig_h // sr_upsample_scale,
                                       orig_w // sr_upsample_scale), mode=mode)
        out = blur(out, sinc_kernel)

    # clamp and round
    lq = torch.clamp((out * 255.0).round(), 0, 255) / 255.

    (gt, gt_usm), lq = paired_random_crop([gt, gt_usm], lq, random_crop_gt_size, sr_upsample_scale)

    return gt, gt_usm, lq
