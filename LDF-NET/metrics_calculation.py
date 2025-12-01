import os
import cv2
import numpy as np
from skimage.metrics import structural_similarity, peak_signal_noise_ratio
from uiqm_utils import getUIQM

def calculate_metrics_ssim_psnr(generated_image_path, ground_truth_image_path, resize_size=(256, 256)):
    generated_image_list = os.listdir(generated_image_path)
    error_list_ssim, error_list_psnr = [], []

    for img in generated_image_list:
        label_img = img
        generated_image = os.path.join(generated_image_path, img)
        ground_truth_image = os.path.join(ground_truth_image_path, label_img)

        # 读取并 resize 图像
        generated_image = cv2.imread(generated_image)
        generated_image = cv2.resize(generated_image, resize_size)

        ground_truth_image = cv2.imread(ground_truth_image)
        ground_truth_image = cv2.resize(ground_truth_image, resize_size)

        # 计算 SSIM 分通道
        ssim_channels = []
        for c in range(3):  # 对 R、G、B 三个通道分别计算 SSIM
            ssim_val, _ = structural_similarity(
                generated_image[:, :, c], ground_truth_image[:, :, c],
                full=True,
                channel_axis=None,  # 单独计算每个通道
                data_range=255
            )
            ssim_channels.append(ssim_val)
        error_ssim = np.mean(ssim_channels)  # 取平均值
        error_list_ssim.append(error_ssim)

        # 计算 PSNR 分通道
        psnr_channels = []
        for c in range(3):  # 对 R、G、B 三个通道分别计算 PSNR
            psnr_val = peak_signal_noise_ratio(
                generated_image[:, :, c], ground_truth_image[:, :, c],
                data_range=255
            )
            psnr_channels.append(psnr_val)
        error_psnr = np.mean(psnr_channels)  # 取平均值
        error_list_psnr.append(error_psnr)

    return np.array(error_list_ssim), np.array(error_list_psnr)

def calculate_UIQM(image_path, resize_size=(256, 256)):
    image_list = os.listdir(image_path)
    uiqms = []

    for img in image_list:
        image = os.path.join(image_path, img)

        image = cv2.imread(image)
        image = cv2.resize(image, resize_size)

        # 计算 UIQM
        uiqms.append(getUIQM(image))
    return np.array(uiqms)
