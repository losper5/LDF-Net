import os
import argparse
import time
import numpy as np
import torch
import torchvision
from torchvision import transforms
from tqdm import tqdm

from model import UWnet
from dataloader import UWNetDataSet
from metrics_calculation import calculate_metrics_ssim_psnr, calculate_UIQM

@torch.no_grad()
def test(config, test_loader, model):
    model.eval()
    for img, _, name in tqdm(test_loader, desc="Testing"):
        img = img.to(config.device)
        output = model(img)
        out_path = os.path.join(config.output_images_path, name[0])
        torchvision.utils.save_image(output, out_path)

def setup(config):
    # 设备
    config.device = "cuda" if torch.cuda.is_available() else "cpu"
    # 模型初始化
    model = UWnet(num_layers=config.num_layers).to(config.device)
    # 加载 checkpoint
    ckpt = torch.load(config.snapshot_path, map_location=config.device)
    if isinstance(ckpt, dict):
        # 如果是 dict，可能包含 state_dict 或直接就是 state_dict
        if 'model_state_dict' in ckpt:
            model.load_state_dict(ckpt['model_state_dict'])
        else:
            model.load_state_dict(ckpt)
    elif isinstance(ckpt, UWnet):
        # 如果直接保存了模型实例，替换掉初始化的 model
        model = ckpt.to(config.device)
    else:
        raise TypeError(f"Unrecognized checkpoint format: {type(ckpt)}")

    # 数据加载
    transform = transforms.Compose([
        transforms.Resize((config.resize, config.resize)),
        transforms.ToTensor()
    ])
    test_ds = UWNetDataSet(config.test_images_path, None, transform, False)
    test_loader = torch.utils.data.DataLoader(
        test_ds, batch_size=config.batch_size, shuffle=False
    )
    print("Test Dataset Reading Completed.")
    return test_loader, model

def main(config):
    # 确保输出目录存在
    os.makedirs(config.output_images_path, exist_ok=True)

    start = time.time()
    test_loader, model = setup(config)
    test(config, test_loader, model)
    print("Total testing time:", time.time() - start)

    if config.calculate_metrics:
        print("-------------------calculating performance metrics---------------------")
        ssim_vals, psnr_vals = calculate_metrics_ssim_psnr(
            config.output_images_path,
            config.label_images_path,
            resize_size=(config.resize, config.resize)
        )
        uiqm_vals = calculate_UIQM(
            config.output_images_path,
            resize_size=(config.resize, config.resize)
        )
        print(f"SSIM on {len(ssim_vals)} samples: {np.mean(ssim_vals):.3f} ± {np.std(ssim_vals):.3f}")
        print(f"PSNR on {len(psnr_vals)} samples: {np.mean(psnr_vals):.3f} ± {np.std(psnr_vals):.3f}")
        print(f"UIQM on {len(uiqm_vals)} samples: {np.mean(uiqm_vals):.3f} ± {np.std(uiqm_vals):.3f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--snapshot_path', type=str,
        default='./Ablation/deep_mfm/model_epoch_240.ckpt',
        help='path to checkpoint (.ckpt) file'
    )
    parser.add_argument(
        '--test_images_path', type=str, default="./data/EUVP/",
        help='path of input images for testing'
    )
    parser.add_argument(
        '--output_images_path', type=str, default='./data/deep_mfm/epoch240',
        help='directory to save generated images'
    )
    parser.add_argument(
        '--batch_size', type=int, default=1,
        help='testing batch size'
    )
    parser.add_argument(
        '--resize', type=int, default=256,
        help='resize dimension for input/output images'
    )
    parser.add_argument(
        '--calculate_metrics', action='store_true', default=False,
        help='whether to compute PSNR/SSIM/UIQM after testing'
    )
    parser.add_argument(
        '--label_images_path', type=str, default="./data/GTr/",
        help='path of ground truth images for metrics'
    )
    parser.add_argument(
        '--num_layers', type=int, default=3,
        help='number of ConvBlock layers in UWnet'
    )
    config = parser.parse_args()
    print("-------------------testing---------------------")
    main(config)