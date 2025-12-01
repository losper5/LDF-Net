import os
import glob
import argparse

import cv2
import torch
from skimage.metrics import structural_similarity, peak_signal_noise_ratio
import libsvm.svmutil as svmutil
from brisque import BRISQUE

svmutil.PRECOMPUTED = 4

def load_tensor(path):
    """加载图像并转换为 [1, C, H, W] 的 float tensor（范围 [0,1]）。"""
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    t   = torch.from_numpy(img).permute(2,0,1).unsqueeze(0).float() / 255.0
    return t

def eval_pair_skimage(ref_path, out_path, resize_size=(256, 256)):
    """
    使用 skimage.metrics 计算 PSNR 和 SSIM。
    - 先将彩色图像 resize 到指定大小
    - 用 multichannel SSIM
    - 将图像转为灰度后计算 PSNR
    """
    # 读取并 resize
    img_out = cv2.imread(out_path, cv2.IMREAD_COLOR)
    img_ref = cv2.imread(ref_path, cv2.IMREAD_COLOR)
    img_out = cv2.resize(img_out, resize_size)
    img_ref = cv2.resize(img_ref, resize_size)

    # 计算 SSIM（多通道）
    s_val, _ = structural_similarity(
        img_out, img_ref,
        full=True,
        multichannel=True,
        channel_axis=-1
    )

    # 转灰度后计算 PSNR
    gray_out = cv2.cvtColor(img_out, cv2.COLOR_BGR2GRAY)
    gray_ref = cv2.cvtColor(img_ref, cv2.COLOR_BGR2GRAY)
    p_val = peak_signal_noise_ratio(gray_out, gray_ref)

    return p_val, s_val

def main(ref_folder, out_root):
    br = BRISQUE(url=False)
    variants = ["epoch63"]  # 可以根据需要修改
    results = {}

    for v in variants:
        folder = os.path.join(out_root, v)
        outs   = sorted(glob.glob(os.path.join(folder, '*.*')))
        if not outs:
            print(f"[Warning] 目录空，跳过: {folder}")
            continue

        sum_psnr    = 0.0
        sum_ssim    = 0.0
        sum_brisque = 0.0
        count       = 0

        for out_path in outs:
            name = os.path.basename(out_path)
            ref_path = os.path.join(ref_folder, name)
            if not os.path.isfile(ref_path):
                print(f"[Warning] 找不到参考图 {ref_path}，跳过")
                continue

            # 使用 skimage.metrics 计算 PSNR & SSIM
            p_val, s_val = eval_pair_skimage(ref_path, out_path)
            # 使用 BRISQUE 计算无参考质量指标
            b_val = br.score(cv2.imread(out_path, cv2.IMREAD_COLOR))

            sum_psnr    += p_val
            sum_ssim    += s_val
            sum_brisque += b_val
            count       += 1

        if count > 0:
            results[v] = {
                'psnr':    sum_psnr    / count,
                'ssim':    sum_ssim    / count,
                'brisque': sum_brisque / count,
                'n':       count
            }

    # 打印平均结果
    print("Model    #Images    Avg_PSNR    Avg_SSIM    Avg_BRISQUE")
    for v, r in results.items():
        print(f"{v:<8}{r['n']:11d}"
              f"{r['psnr']:12.3f}{r['ssim']:12.3f}{r['brisque']:14.3f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ref_folder", type=str, default="./data/reference-890/",
                        help="参考图像文件夹")
    parser.add_argument("--out_root",  type=str, default="./data/output/",
                        help="输出结果根目录")
    args = parser.parse_args()
    main(args.ref_folder, args.out_root)
