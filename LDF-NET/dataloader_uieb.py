import os
import torch
from PIL import Image
import torchvision.transforms.functional as TF
from torchvision import transforms

def get_image_list(raw_image_path, clear_image_path, is_train):
    image_list = []
    raw_files = os.listdir(raw_image_path)
    for fname in raw_files:
        raw_full = os.path.join(raw_image_path, fname)
        if is_train:
            clear_full = os.path.join(clear_image_path, fname)
            image_list.append([raw_full, clear_full, fname])
        else:
            image_list.append([raw_full, None, fname])
    return image_list

class UWNetDataSet(torch.utils.data.Dataset):
    def __init__(self, raw_image_path, clear_image_path, transform, is_train=False):
        self.raw_image_path   = raw_image_path
        self.clear_image_path = clear_image_path
        self.is_train         = is_train
        self.transform        = transform
        self.crop_size = None
        self.items     = get_image_list(raw_image_path, clear_image_path, is_train)

    def set_crop_size(self, size):
        """
        设置 RandomCrop 的输出尺寸 (h, w)。
        training.py 里会调用： train_ds.set_crop_size((cfg.crop_size, cfg.crop_size))
        """
        self.crop_size = size

    def __getitem__(self, idx):
        raw_path, clear_path, name = self.items[idx]
        raw_img = Image.open(raw_path).convert('RGB')

        if self.is_train:
            clear_img = Image.open(clear_path).convert('RGB')

            if self.crop_size is not None:
                th, tw = self.crop_size
                w, h   = raw_img.size
                if h < th or w < tw:
                    raw_img   = raw_img.resize((tw, th), Image.BILINEAR)
                    clear_img = clear_img.resize((tw, th), Image.BILINEAR)
                i, j, hh, ww = transforms.RandomCrop.get_params(raw_img, output_size=self.crop_size)
                raw_img   = TF.crop(raw_img,   i, j, hh, ww)
                clear_img = TF.crop(clear_img, i, j, hh, ww)

            raw_t   = self.transform(raw_img)
            clear_t = self.transform(clear_img)
            return raw_t, clear_t, name

        else:
            raw_t = self.transform(raw_img)
            return raw_t, raw_t, name

    def __len__(self):
        return len(self.items)
