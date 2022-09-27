import math
import os

import numpy as np
from basicsr.data.degradations import circular_lowpass_kernel, random_mixed_kernels
from basicsr.utils import DiffJPEG, USMSharp
from numpy.typing import NDArray
from PIL import Image
from torch import Tensor
from torch.nn import functional as F

from data.degradations import apply_real_esrgan_degradations
from utils.utils_video import img2tensor


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




def degrade_imgs(src_folder: str, dst_folder: str, degrade_scale: float, start_size: int) -> None:
    src_img_filenames = os.listdir(src_folder)
    jpeg_simulator = DiffJPEG()
    usm_sharpener = USMSharp()
    for src_img_filename in src_img_filenames:
        src_img = Image.open(os.path.join(src_folder, src_img_filename))

        src_tensor = img2tensor(np.array(src_img), bgr2rgb=False,
                                float32=True).unsqueeze(0) / 255.0
        orig_h, orig_w = src_tensor.size()[2:4]
        print("SRC TENSOR orig size: ", src_tensor.size())
        if orig_h != start_size or orig_w != start_size:
            src_tensor = F.interpolate(src_tensor, size=(start_size, start_size), mode='bicubic')
            print("SRC TENSOR new size: ", src_tensor.size())

        blur_kernel1, blur_kernel2, sinc_kernel = _decide_kernels()
        (src, src_sharp, degraded_img) = apply_real_esrgan_degradations(
            src_tensor,
            blur_kernel1=Tensor(blur_kernel1).unsqueeze(0),
            blur_kernel2=Tensor(blur_kernel2).unsqueeze(0),
            second_blur_prob=0.4,
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
            jpeg_simulator=jpeg_simulator,
            random_crop_gt_size=start_size,
            sr_upsample_scale=1,
            usm_sharpener=usm_sharpener
        )

        # print(src.size())
        # print(src_sharp.size())
        # print(degraded_img.size())
        # print(torch.max(src))
        # print(torch.max(src_sharp))
        # print(torch.max(degraded_img))
        # print(torch.min(src))
        # print(torch.min(src_sharp))
        # print(torch.min(degraded_img))
        # Image.fromarray((src[0] * 255.0).permute(1, 2, 0).cpu().numpy().astype(np.uint8)).save(
        #     "/home/cll/Desktop/TEST_IMAGE1.png")
        # Image.fromarray((src_sharp[0] * 255.0).permute(
        #     1, 2, 0).cpu().numpy().astype(np.uint8)).save(
        #         "/home/cll/Desktop/TEST_IMAGE2.png")

        Image.fromarray((degraded_img[0] * 255.0).permute(
            1, 2, 0).cpu().numpy().astype(np.uint8)).save(
                os.path.join(dst_folder, src_img_filename))
        print("SAVED %s: " % src_img_filename)

        # Image.fromarray((src_tensor[0] * 255.0).permute(
        #     1, 2, 0).cpu().numpy().astype(np.uint8)).save(
        #         os.path.join(dst_folder, src_img_filename))
        # print("SAVED %s: " % src_img_filename)


if __name__ == "__main__":
    SRC_FOLDER = "/home/cll/Desktop/sr_test_GT_HQ"
    OUTPUT_RESOLUTION_SCALE = 1
    DST_FOLDER = "/home/cll/Desktop/sr_test_degraded_LQ_512"
    # DST_FOLDER = "/home/cll/Desktop/sr_test_GT_512"
    os.makedirs(DST_FOLDER, exist_ok=True)

    degrade_imgs(SRC_FOLDER, DST_FOLDER, OUTPUT_RESOLUTION_SCALE, 512)
