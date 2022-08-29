import os
import random

import numpy as np
import torch.utils.data as data
import utils.image as img_util

from augs import sr_degradations as sr


class BlindSRDataset(data.Dataset):
    def __init__(
        self,
        hq_data_dir: str,
        sr_scale: int,
        degradation_type: str,
        is_training: bool = True,
        n_channels: int = 3,
        hq_size: int = None,
        shuffle_prob: int = 0.1,
        use_sharp: bool = False,
        hq_patch_size: int = None,
        lq_patch_size: int = 64,
    ):
        self.n_channels = n_channels
        self.sr_scale = sr_scale
        self.shuffle_prob = shuffle_prob
        self.use_sharp = use_sharp
        self.degradation_type = degradation_type
        self.is_training = is_training
        self.hq_patch_size = hq_patch_size if hq_patch_size \
            else lq_patch_size * sr_scale
        self.lq_patch_size = lq_patch_size

        self.hq_img_paths = img_util.get_image_paths(hq_data_dir)
        assert self.hq_img_paths, 'Error: H path is empty.'

    def __getitem__(self, index):

        lq_img_path = None

        # ------------------------------------
        # get H image
        # ------------------------------------
        hq_img_path = self.hq_img_paths[index]
        hq_img = img_util.imread_uint(hq_img_path, self.n_channels)
        img_name, ext = os.path.splitext(os.path.basename(hq_img_path))
        H, W, C = hq_img.shape

        if H < self.hq_patch_size or W < self.hq_patch_size:
            hq_img = np.tile(np.random.randint(0, 256, size=[1, 1, self.n_channels],
                                               dtype=np.uint8),
                             (self.hq_patch_size, self.hq_patch_size, 1))

        # ------------------------------------
        # if train, get L/H patch pair
        # ------------------------------------
        # print("IS_TRAIN: ", str(self.is_training))
        if self.is_training:

            H, W, C = hq_img.shape

            rnd_h_H = random.randint(0, max(0, H - self.hq_patch_size))
            rnd_w_H = random.randint(0, max(0, W - self.hq_patch_size))
            hq_img = hq_img[rnd_h_H:rnd_h_H + self.hq_patch_size,
                            rnd_w_H:rnd_w_H + self.hq_patch_size, :]

            if 'face' in img_name:
                mode = random.choice([0, 4])
                hq_img = img_util.augment_img(hq_img, mode=mode)
            else:
                mode = random.randint(0, 7)
                hq_img = img_util.augment_img(hq_img, mode=mode)

            hq_img = img_util.uint2single(hq_img)
            if self.degradation_type == 'bsrgan':
                lq_img, hq_img = sr.degradation_bsrgan(hq_img, self.sr_scale,
                                                       lq_patch_size=self.lq_patch_size,
                                                       isp_model=None)
            elif self.degradation_type == 'bsrgan_plus':
                lq_img, hq_img = sr.degradation_bsrgan_plus(hq_img, self.sr_scale,
                                                            shuffle_prob=self.shuffle_prob,

                                                            use_sharp=self.use_sharp,
                                                            lq_patch_size=self.lq_patch_size)

        else:
            hq_img = img_util.uint2single(hq_img)
            if self.degradation_type == 'bsrgan':
                lq_img, hq_img = sr.degradation_bsrgan(hq_img, self.sr_scale,
                                                       lq_patch_size=self.lq_patch_size,
                                                       isp_model=None)
            elif self.degradation_type == 'bsrgan_plus':
                lq_img, hq_img = sr.degradation_bsrgan_plus(hq_img, self.sr_scale,
                                                            shuffle_prob=self.shuffle_prob,
                                                            use_sharp=self.use_sharp,
                                                            lq_patch_size=self.lq_patch_size)

        # ------------------------------------
        # L/H pairs, HWC to CHW, numpy to tensor
        # ------------------------------------
        hq_img, lq_img = img_util.single2tensor3(hq_img), img_util.single2tensor3(lq_img)

        if lq_img_path is None:
            lq_img_path = hq_img_path

        return {
            'lq_img': lq_img,
            'hq_img': hq_img,
            'lq_img_path': lq_img_path,
            'hq_img_path': hq_img_path
        }

    def __len__(self):
        return len(self.hq_img_paths)
