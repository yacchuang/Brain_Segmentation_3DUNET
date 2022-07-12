#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 12  2022

@author: kurtlab
"""

import os
from PIL import Image
from torch.utils.data import Dataset
import SimpleITK as sitk
import nibabel as nib
from LoadVisualNIFTI import read_img_nii, read_img_sitk, np_BrainImg, np_PFMaskImg
import numpy as np


class BrainDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.images = os.listdir(image_dir)


    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        img_path = os.path.join(self.image_dir, self.images[index])
        mask_path = os.path.join(self.mask_dir, self.images[index].replace("T1_", "PFseg_"))  # Differences of filename between images and masks
        read_img_sitk(img_path)
        read_img_sitk(mask_path)
        BrainImg = read_img_nii(img_path)
        PFMaskImg = read_img_nii(mask_path)

        PFMask = sitk.GetImageFromArray(PFMaskImg)
        BrainT1 = sitk.GetImageFromArray(BrainImg)
        image = sitk.GetArrayFromImage(BrainT1)
        mask = sitk.GetArrayFromImage(PFMask)


        # image = np.array(Image.open(img_path).convert("RGB"))
        # mask = np.array(Image.open(mask_path).convert("L"), dtype=np.float32)
        mask[mask == 255.0] = 1.0

        if self.transform is not None:
            augmentations = self.transform(image=image, mask=mask)
            image = augmentations["image"]
            mask = augmentations["mask"]

        return image, mask