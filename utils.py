#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul  3 10:08:04 2022

@author: kurtlab
"""

import torch
import torchvision
from BrainDataset import BrainDataset
from torch.utils.data import DataLoader
import nibabel as nib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def save_checkpoint(state, filename = "PFcheckpoint_EPO5.pth.tar"):
    print("=> Saving checkpoint")
    torch.save(state, filename)
    
def load_checkpoint(checkpoint, model):
    print("=> Loading checkpoint")
    model.load_state_dict(checkpoint["state_dict"])
    
def get_loaders(
    train_dir,
    train_maskdir,
    val_dir,
    val_maskdir,
    batch_size,
    aug,
    # train_transform,
    # val_transform,
    num_workers=4,
    pin_memory=True,
        
):
    train_ds = BrainDataset(
        image_dir=train_dir,
        mask_dir=train_maskdir,
        transform=aug,
        # transform=train_transform,
    )
    
    
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=True,
    )
    
    
    val_ds = BrainDataset(
        image_dir=val_dir,
        mask_dir=val_maskdir,
        transform=aug,
        # transform=val_transform,
    )
        
    
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=False,
    )
    
    
    return train_loader, val_loader


def check_accuracy(loader, model, device="cuda"):
    num_correct = 0
    num_pixels = 0
    dice_score = 0    # For multiclass seg, probably a better score
    model.eval()
    
    
    with torch.no_grad():
        for x, y in loader:
            x = x.float().unsqueeze(1).to(device)
            y = y.float().unsqueeze(1).to(device)  # label doesn't have color
            preds = torch.sigmoid(model(x))
            preds = (preds > 0.5).float()
            num_correct += (preds == y).sum()
            num_pixels += torch.numel(preds)
            dice_score += (2 * (preds * y).sum()) / (
                (preds + y).sum() + 1e-8
                
            )
            # # Plot and compare train and prediction images in 2D
            # plt.figure()
            # plt.imshow(x[0, 0, 64, :, :], cmap='gray')
            # plt.figure()
            # plt.imshow(preds[0, 0, 64, :, :], cmap='gray', alpha=0.7)
            # plt.show()

    print(f"Got {num_correct}/{num_pixels} with acc {num_correct}/{num_pixels*100:.2f}")
    
    print(f"Dice score: {dice_score/len(loader)}")
    
    model.train()
    
    
    
def save_predictions(
    loader, model, folder="/Users/kurtlab/Documents/GitHub/Brain_Segmentation/saved_BrainSegImages", device="cuda"
):
    model.eval()
    for idx, (x, y) in enumerate(loader):
        x = x.float().unsqueeze(1).to(device=device)
        with torch.no_grad():
            preds = torch.sigmoid(model(x))
            preds = (preds > 0.5).float()
        torchvision.utils.save_image(preds[0, 0, 64, :, :], f"{folder}/pred_{idx}.png")
        torchvision.utils.save_image(y.float().unsqueeze(1)[0, 0, 64, :, :], f"{folder}/train_{idx}.png")
        # Convert numpy array to NIFTI
        nifti = nib.Nifti1Image(preds, None)
        # nifti.get_data_dtype() = seg.dtype
        # Save segmentation to disk
        nib.save(nifti, f"{folder}/prediction_{idx}.nii.gz")

    model.train()