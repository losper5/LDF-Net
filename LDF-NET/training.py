import os
import argparse
import numpy as np
import torch
import torchvision
from torch.nn import Module
from torchvision import transforms
from dataclasses import dataclass
from tqdm.autonotebook import tqdm, trange

import wandb
from dataloader import UWNetDataSet
from metrics_calculation import calculate_metrics_ssim_psnr, calculate_UIQM
from model import UWnet
from combined_loss import combinedloss

@dataclass
class Trainer:
    model: Module
    opt: torch.optim.Optimizer
    loss: Module

    @torch.enable_grad()
    def train(self, train_loader, config, test_loader=None):
        device = config.device

        # 训练循环
        for epoch in trange(0, config.num_epochs, desc="[Full Loop]", leave=False):
            # 确保训练模式
            self.model.train()

            # 学习率衰减
            if epoch > 1 and epoch % config.step_size == 0:
                for g in self.opt.param_groups:
                    g['lr'] *= 0.7

            # 1) 训练一个 epoch
            total_loss = 0.0
            mse_loss   = 0.0
            vgg_loss   = 0.0
            for inp, label, _ in tqdm(train_loader, desc="[Train]", leave=False):
                inp, label = inp.to(device), label.to(device)
                self.opt.zero_grad()
                out = self.model(inp)
                loss, mse_l, vgg_l = self.loss(out, label)
                loss.backward()
                self.opt.step()
                total_loss += loss.item()
                mse_loss   += mse_l.item()
                vgg_loss   += vgg_l.item()

            # 记录训练损失
            wandb.log({
                "[Train] Total Loss": total_loss / len(train_loader),
                "[Train] Primary Loss": mse_loss / len(train_loader),
                "[Train] VGG Loss": vgg_loss / len(train_loader),
            }, commit=True)

            eval_end = config.eval_end or config.num_epochs
            if (config.test
                and (epoch + 1) >= config.eval_start
                and (epoch + 1) <= eval_end
                and ((epoch + 1 - config.eval_start) % config.eval_freq == 0)
            ):
                # 切到 eval 模式
                self.model.eval()
                UIQM, SSIM, PSNR = self.eval(config, test_loader, self.model)
                # 切回 train 模式
                self.model.train()
                # 记录一次验证指标
                wandb.log({
                    "[Test] Epoch": epoch + 1,
                    "[Test] UIQM": np.mean(UIQM),
                    "[Test] SSIM": np.mean(SSIM),
                    "[Test] PSNR": np.mean(PSNR),
                }, commit=True)

            # 3) 打印进度
            if epoch % config.print_freq == 0:
                print(f"epoch [{epoch}/{config.num_epochs}] "
                      f"Total {total_loss/len(train_loader):.4f}, "
                      f"MSE {mse_loss/len(train_loader):.4f}, "
                      f"VGG {vgg_loss/len(train_loader):.4f}")

            # 4) 保存 checkpoint
            os.makedirs(config.snapshots_folder, exist_ok=True)
            if epoch % config.snapshot_freq == 0:
                ckpt_path = os.path.join(
                    config.snapshots_folder,
                    f"model_epoch_{epoch}.ckpt"
                )
                torch.save({
                    'epoch': epoch + 1,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.opt.state_dict(),
                }, ckpt_path)

    @torch.no_grad()
    def eval(self, config, test_loader, model):
        # 清空旧的可视化输出
        out_dir = config.output_images_path
        if os.path.isdir(out_dir):
            for f in os.listdir(out_dir):
                fp = os.path.join(out_dir, f)
                if os.path.isfile(fp):
                    os.remove(fp)
        else:
            os.makedirs(out_dir, exist_ok=True)

        # 推理并保存图像
        for img, _, name in test_loader:
            img = img.to(config.device)
            gen = model(img)
            save_path = os.path.join(out_dir, name[0])
            torchvision.utils.save_image(gen, save_path)

        # 计算指标
        ssim_vals, psnr_vals = calculate_metrics_ssim_psnr(
            config.output_images_path,
            config.GTr_test_images_path,
            resize_size=(config.resize, config.resize)
        )
        uiqm_vals = calculate_UIQM(
            config.output_images_path,
            resize_size=(config.resize, config.resize)
        )
        return uiqm_vals, ssim_vals, psnr_vals

def setup(config):
    # 设备
    config.device = "cuda" if torch.cuda.is_available() else "cpu"
    # 模型
    model = UWnet(num_layers=config.num_layers, base_ch=64).to(config.device)
    # 数据
    tf = transforms.Compose([
        # transforms.Resize((config.resize, config.resize)),
        transforms.ToTensor(),
    ])
    train_ds = UWNetDataSet(
        config.input_images_path,
        config.label_images_path,
        tf, True
    )
    train_dl = torch.utils.data.DataLoader(
        train_ds,
        batch_size=config.train_batch_size,
        shuffle=False
    )
    # 损失与优化器
    loss = combinedloss(config)
    opt  = torch.optim.Adam(model.parameters(), lr=config.lr,betas=(0.9, 0.999), eps=1e-8, weight_decay=0 )
    trainer = Trainer(model, opt, loss)

    # 测试集
    if config.test:
        test_ds = UWNetDataSet(
            config.test_images_path,
            None,
            tf, False
        )
        test_dl = torch.utils.data.DataLoader(
            test_ds,
            batch_size=config.test_batch_size,
            shuffle=False
        )
        return train_dl, test_dl, model, trainer

    return train_dl, None, model, trainer

def training(config):
    wandb.init(project="underwater_image_enhancement_UWNet")
    wandb.config.update(config, allow_val_change=True)
    config = wandb.config

    train_dl, test_dl, model, trainer = setup(config)
    trainer.train(train_dl, config, test_dl)
    print("=== Training complete! ===")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_images_path',    type=str,   default="./data/EUVP/")
    parser.add_argument('--label_images_path',    type=str,   default="./data/GTr/")
    parser.add_argument('--test_images_path',     type=str,   default="./data/EUVP/")
    parser.add_argument('--GTr_test_images_path', type=str,   default="./data/GTr/")
    parser.add_argument('--test',                 action='store_true', default=True)
    parser.add_argument('--lr',                   type=float, default=0.0002)
    parser.add_argument('--eval_freq',    type=int,   default=10,help='每隔多少个 epoch 在 test_loader 上 eval 一次')
    parser.add_argument('--eval_start',   type=int,   default=230)
    parser.add_argument('--eval_end',     type=int,   default=None, help='结束 eval 的 epoch 1-based默认到最后一轮')
    parser.add_argument('--step_size',            type=int,   default=400)
    parser.add_argument('--num_epochs',           type=int,   default=250)
    parser.add_argument('--train_batch_size',     type=int,   default=16)
    parser.add_argument('--test_batch_size',      type=int,   default=1)
    parser.add_argument('--resize',               type=int,   default=256)
    parser.add_argument('--print_freq',           type=int,   default=1)
    parser.add_argument('--snapshot_freq',        type=int,   default=10)
    parser.add_argument('--snapshots_folder',     type=str,   default="./Ablation/")
    parser.add_argument('--output_images_path',   type=str,   default="./data/")
    parser.add_argument('--num_layers',           type=int,   default=3)

    config = parser.parse_args()
    os.makedirs(config.snapshots_folder,   exist_ok=True)
    os.makedirs(config.output_images_path, exist_ok=True)

    training(config)