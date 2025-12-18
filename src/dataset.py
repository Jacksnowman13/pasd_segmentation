import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image

class PASDDataset(Dataset): 
    def __init__(self, img_dir, mask_dir, transform=None):
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.transform = transform

        self.image_files = sorted([
            f for f in os.listdir(img_dir)
            if f.lower().endswith((".jpg", ".jpeg", ".png"))
        ])
        self.mask_files = sorted([
            f for f in os.listdir(mask_dir)
            if f.lower().endswith((".png",)) 
        ])
        assert len(self.image_files) == len(self.mask_files), "Checks for image/mask count are same" 

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        mask_name = self.mask_files[idx]

        img_path = os.path.join(self.img_dir, img_name)
        mask_path = os.path.join(self.mask_dir, mask_name)

        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        if img is None:
            raise FileNotFoundError("ERROR! BEN")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        mask = np.array(Image.open(mask_path), dtype=np.uint8)

        if self.transform is not None:
            aug = self.transform(image=img, mask=mask)
            img = aug["image"]
            mask = aug["mask"]

        img = img.astype(np.float32) / 255.0
        img = np.transpose(img, (2, 0, 1)) 
        img_tensor = torch.from_numpy(img)
        mask_tensor = torch.from_numpy(mask.astype(np.int64))

        return img_tensor, mask_tensor
