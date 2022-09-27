import math
import numpy as np
import random
import torch
import torch.utils.data as data
import utils.utils_image as util
from basicsr.data.degradations import circular_lowpass_kernel, random_mixed_kernels
from basicsr.utils import DiffJPEG, USMSharp
from numpy.typing import NDArray
from PIL import Image
from utils.utils_video import img2tensor
from torch import Tensor

from data.degradations import apply_real_esrgan_degradations

class DatasetSR(data.Dataset):
    '''
    # -----------------------------------------
    # Get L/H for SISR.
    # If only "paths_H" is provided, sythesize bicubicly downsampled L on-the-fly.
    # -----------------------------------------
    # e.g., SRResNet
    # -----------------------------------------
    '''

    def __init__(self, opt):
        super(DatasetSR, self).__init__()
        self.opt = opt
        self.n_channels = opt['n_channels'] if opt['n_channels'] else 3
        self.sf = opt['scale'] if opt['scale'] else 4
        self.patch_size = self.opt['H_size'] if self.opt['H_size'] else 96
        self.L_size = self.patch_size // self.sf

        # ------------------------------------
        # get paths of L/H
        # ------------------------------------
        self.paths_H = util.get_image_paths(opt['dataroot_H'])
        self.paths_L = util.get_image_paths(opt['dataroot_L'])

        assert self.paths_H, 'Error: H path is empty.'
        if self.paths_L and self.paths_H:
            assert len(self.paths_L) == len(self.paths_H), 'L/H mismatch - {}, {}.'.format(len(self.paths_L), len(self.paths_H))

        self.jpeg_simulator = DiffJPEG()
        self.usm_sharpener = USMSharp()

    blur_kernel_list1 = ['iso', 'aniso', 'generalized_iso',
                         'generalized_aniso', 'plateau_iso', 'plateau_aniso']
    blur_kernel_list2 = ['iso', 'aniso', 'generalized_iso',
                         'generalized_aniso', 'plateau_iso', 'plateau_aniso']
    blur_kernel_prob1 = [0.45, 0.25, 0.12, 0.03, 0.12, 0.03]
    blur_kernel_prob2 = [0.45, 0.25, 0.12, 0.03, 0.12, 0.03]
    kernel_size = 21
    blur_sigma1 = [0.05, 0.2]
    blur_sigma2 = [0.05, 0.1]
    betag_range1 = [0.7, 1.3]
    betag_range2 = [0.7, 1.3]
    betap_range1 = [0.7, 1.3]
    betap_range2 = [0.7, 1.3]

    def _decide_kernels(self) -> NDArray:
        blur_kernel1 = random_mixed_kernels(
            self.blur_kernel_list1,
            self.blur_kernel_prob1,
            self.kernel_size,
            self.blur_sigma1,
            self.blur_sigma1, [-math.pi, math.pi],
            self.betag_range1,
            self.betap_range1,
            noise_range=None
        )
        blur_kernel2 = random_mixed_kernels(
            self.blur_kernel_list2,
            self.blur_kernel_prob2,
            self.kernel_size,
            self.blur_sigma2,
            self.blur_sigma2, [-math.pi, math.pi],
            self.betag_range2,
            self.betap_range2,
            noise_range=None
        )
        if self.kernel_size < 13:
            omega_c = np.random.uniform(np.pi / 3, np.pi)
        else:
            omega_c = np.random.uniform(np.pi / 5, np.pi)
        sinc_kernel = circular_lowpass_kernel(omega_c, self.kernel_size, pad_to=21)
        return (blur_kernel1, blur_kernel2, sinc_kernel)

    def __getitem__(self, index):

        L_path = None
        # ------------------------------------
        # get H image
        # ------------------------------------
        H_path = self.paths_H[index]
        img_H = util.imread_uint(H_path, self.n_channels)
        img_H = util.uint2single(img_H)

        # ------------------------------------
        # modcrop
        # ------------------------------------
        img_H = util.modcrop(img_H, self.sf)

        # ------------------------------------
        # get L image
        # ------------------------------------
        if self.paths_L:
            # --------------------------------
            # directly load L image
            # --------------------------------
            L_path = self.paths_L[index]
            img_L = util.imread_uint(L_path, self.n_channels)
            img_L = util.uint2single(img_L)

        else:
            # --------------------------------
            # sythesize L image via matlab's bicubic
            # --------------------------------
            H, W = img_H.shape[:2]
            img_L = util.imresize_np(img_H, 1 / self.sf, True)

        src_tensor = img2tensor(img_L.copy(), bgr2rgb=False,
                                float32=True).unsqueeze(0)

        blur_kernel1, blur_kernel2, sinc_kernel = self._decide_kernels()
        (img_L_2, sharp_img_L, degraded_img_L) = apply_real_esrgan_degradations(
            src_tensor,
            blur_kernel1=Tensor(blur_kernel1).unsqueeze(0),
            blur_kernel2=Tensor(blur_kernel2).unsqueeze(0),
            second_blur_prob=0.2,
            sinc_kernel=Tensor(sinc_kernel).unsqueeze(0),
            resize_prob1=[0.2, 0.7, 0.1],
            resize_prob2=[0.3, 0.4, 0.3],
            resize_range1=[0.9, 1.1],
            resize_range2=[0.9, 1.1],
            gray_noise_prob1=0.2,
            gray_noise_prob2=0.2,
            gaussian_noise_prob1=0.2,
            gaussian_noise_prob2=0.2,
            noise_range=[0.01, 0.2],
            poisson_scale_range=[0.05, 0.45],
            jpeg_compression_range1=[85, 100],
            jpeg_compression_range2=[85, 100],
            jpeg_simulator=self.jpeg_simulator,
            random_crop_gt_size=256,
            sr_upsample_scale=1,
            usm_sharpener=self.usm_sharpener
        )
        # Image.fromarray((degraded_img_L[0] * 255).permute(
        #     1, 2, 0).cpu().numpy().astype(np.uint8)).save(
        #     "/home/cll/Desktop/degraded_L.png")
        # Image.fromarray((img_L * 255).astype(np.uint8)).save(
        #     "/home/cll/Desktop/img_L.png")
        # Image.fromarray((img_L_2[0] * 255).permute(
        #     1, 2, 0).cpu().numpy().astype(np.uint8)).save(
        #     "/home/cll/Desktop/img_L_2.png")
        # exit()

        # ------------------------------------
        # if train, get L/H patch pair
        # ------------------------------------
        if self.opt['phase'] == 'train':

            H, W, C = img_L.shape

            # --------------------------------
            # randomly crop the L patch
            # --------------------------------
            rnd_h = random.randint(0, max(0, H - self.L_size))
            rnd_w = random.randint(0, max(0, W - self.L_size))
            img_L = img_L[rnd_h:rnd_h + self.L_size, rnd_w:rnd_w + self.L_size, :]

            # --------------------------------
            # crop corresponding H patch
            # --------------------------------
            rnd_h_H, rnd_w_H = int(rnd_h * self.sf), int(rnd_w * self.sf)
            img_H = img_H[rnd_h_H:rnd_h_H + self.patch_size, rnd_w_H:rnd_w_H + self.patch_size, :]

            # --------------------------------
            # augmentation - flip and/or rotate + RealESRGAN modified degradations
            # --------------------------------
            mode = random.randint(0, 7)
            img_L, img_H = util.augment_img(img_L, mode=mode), util.augment_img(img_H, mode=mode)


        # ------------------------------------
        # L/H pairs, HWC to CHW, numpy to tensor
        # ------------------------------------
        img_H, img_L = util.single2tensor3(img_H), util.single2tensor3(img_L)

        if L_path is None:
            L_path = H_path

        return {'L': img_L, 'H': img_H, 'L_path': L_path, 'H_path': H_path}

    def __len__(self):
        return len(self.paths_H)
