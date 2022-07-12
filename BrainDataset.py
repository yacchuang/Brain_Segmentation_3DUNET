#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 12  2022

@author: kurtlab
"""

import os
from PIL import Image
from torch.utils.data import Dataset
from LoadVisualNIFTI import read_img_nii, read_img_sitk, np_BrainImg, np_PFMaskImg
import numpy as np


class BrainDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None):
        read_img_sitk(self)
        read_img_nii(self)
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.images = os.listdir(image_dir)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        img_path = os.path.join(self.image_dir, self.images[index])
        mask_path = os.path.join(self.mask_dir, self.images[index].replace("T1_", "PFseg_"))  # Differences of filename between images and masks

        sitk_PFMaskImg2 = sitk.GetImageFromArray(np_PFMaskImg)
        sitk_BrainImg2 = sitk.GetImageFromArray(np_BrainImg)


        # image = np.array(Image.open(img_path).convert("RGB"))
        # mask = np.array(Image.open(mask_path).convert("L"), dtype=np.float32)
        mask[mask == 255.0] = 1.0

        if self.transform is not None:
            augmentations = self.transform(image=image, mask=mask)
            sitk_BrainImg2 = augmentations["image"]
            sitk_PFMaskImg2 = augmentations["mask"]

        return sitk_BrainImg2, sitk_PFMaskImg2