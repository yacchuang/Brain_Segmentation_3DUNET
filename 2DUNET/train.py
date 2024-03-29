#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul  1 16:42:53 2022

@author: kurtlab
"""

import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
from UNETmodel import UNET

import matplotlib.pyplot as plt
plt.switch_backend('TKAgg')

from utils import (
    load_checkpoint,
    save_checkpoint,
    get_loaders,
    check_accuracy,
    save_predictions_as_imgs,
) 

# Hyperparameters etc.
LEARNING_RATE = 1e-4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 16 # CAN INCREASE
NUM_EPOCHS = 3
NUM_WORKERS = 1
IMAGE_HEIGHT = 256 # ORIGINALLY 1280
IMAGE_WIDTH = 256 # ORIGINALLY 1918
PIN_MEMORY = True
LOAD_MODEL = False  # True
# Images directory
TRAIN_IMAGE_DIR = "/Volumes/Kurtlab/Chiari_Morpho_Segmentation/Segmentation/BrainSeg/BrainMRI_train"
TRAIN_MASK_DIR = "/Volumes/Kurtlab/Chiari_Morpho_Segmentation/Segmentation/BrainSeg/PFMask_train"
VAL_IMG_DIR = "/Volumes/Kurtlab/Chiari_Morpho_Segmentation/Segmentation/BrainSeg/BrainMRI_val"
VAL_MASK_DIR = "/Volumes/Kurtlab/Chiari_Morpho_Segmentation/Segmentation/BrainSeg/PFMask_val"


def train_fn(loader, model, optimizer, loss_fn, scaler):
    loop = tqdm(loader)
    
    for batch_idx, (data, targets) in enumerate(loop):
        data = data.to(device=DEVICE)
        targets = targets.float().unsqueeze(1).to(device=DEVICE)
        
        # forward
        with torch.cuda.amp.autocast():
            predictions = model(data)
            loss = loss_fn(predictions, targets)
            
        # backward
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        # update tqdm loop
        loop.set_postfix(loss=loss.item())
        
        
        
        

def main():
    train_transform = A.Compose(
        [
            A.Resize(height = IMAGE_HEIGHT, width = IMAGE_WIDTH),
            A.Rotate(limit = 35, p = 1.0),
            A.HorizontalFlip(p = 0.5),
            A.VerticalFlip(p = 1.0),
            A.Normalize(
                mean = [0.0, 0.0, 0.0],
                std = [1.0, 1.0, 1.0],
                max_pixel_value = 255.0,
            ),
            ToTensorV2(),
        ],
    )
    
    
    val_transform = A.Compose(
        [
            A.Resize(height = IMAGE_HEIGHT, width = IMAGE_WIDTH),
            A.Normalize(
                mean = [0.0, 0.0, 0.0],
                std = [1.0, 1.0, 1.0],
                max_pixel_value = 255.0,
            ),
            ToTensorV2(),
        ],
    )
    
    
    model = UNET(in_channels=3, out_channels=1).to(DEVICE)  # For multiclass segmentation, out_channels can increase
    loss_fn = nn.BCEWithLogitsLoss() # For multiclass seg: cross entropy loss
    optimizer = optim.Adam(model.parameters(), lr = LEARNING_RATE)
    
    train_loader, val_loader = get_loaders(
        TRAIN_IMAGE_DIR,
        TRAIN_MASK_DIR,
        VAL_IMG_DIR,
        VAL_MASK_DIR,
        BATCH_SIZE,
        train_transform,
        val_transform,
        NUM_WORKERS,
        PIN_MEMORY,
        
    )
    
    if LOAD_MODEL:
        load_checkpoint(torch.load("my_checkpoint.pth.tar"), model)
    
    check_accuracy(val_loader, model, device=DEVICE)
    scaler = torch.cuda.amp.GradScaler()
    
    
    for epoch in range(NUM_EPOCHS):
        train_fn(train_loader, model, optimizer, loss_fn, scaler)
    
        # save model
        checkpoint = {
            "state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict(),
        }
        save_checkpoint(checkpoint)
        
        # check accuracy
        check_accuracy(val_loader, model, device=DEVICE)
        
        # print some examples to a folder
        save_predictions_as_imgs(
            val_loader, model, folder="/Users/kurtlab/Documents/GitHub/Brain_Segmentation/saved_BrainSegImages", device=DEVICE
        )
    
    
    
    
if __name__ == "__main__":
    main()
