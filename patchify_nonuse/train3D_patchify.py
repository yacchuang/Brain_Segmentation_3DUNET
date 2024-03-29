#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul  1 16:42:53 2022

@author: kurtlab
"""

import torch
from volumentations import *
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
from UNETmodel3D import UNET

import matplotlib.pyplot as plt
plt.switch_backend('TKAgg')
from BrainDataset_patchify import get_augmentation
from patchify import patchify, unpatchify
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
BATCH_SIZE = 1 # CAN INCREASE
NUM_EPOCHS = 2
NUM_WORKERS = 0
patch_size = (64, 64, 64)
n_classes = 4
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
        data = data.float().unsqueeze(1).to(device=DEVICE)
        # data = patchify(data, (64, 64, 64), step=64)
        targets = targets.float().unsqueeze(1).to(device=DEVICE)
        # targets = patchify(targets, (64, 64, 64), step=64)

        # plt.imshow(data[1,2,3,:,:,32])

        # input_img = np.reshape(data, (-1, data.shape[3], data.shape[4], data.shape[5]))
        # input_mask = np.reshape(targets, (-1, targets.shape[3], targets.shape[4], targets.shape[5]))

        # print(input_img.shape)
        
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
    aug = get_augmentation(patch_size)
    '''
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
    '''
    
    model = UNET(in_channels=1, out_channels=1).to(DEVICE)  # For multiclass segmentation, out_channels can increase
    loss_fn = nn.BCEWithLogitsLoss() # For multiclass seg: cross entropy loss
    optimizer = optim.Adam(model.parameters(), lr = LEARNING_RATE)
    
    train_loader, val_loader = get_loaders(
        TRAIN_IMAGE_DIR,
        TRAIN_MASK_DIR,
        VAL_IMG_DIR,
        VAL_MASK_DIR,
        BATCH_SIZE,
        aug,
        NUM_WORKERS,
        PIN_MEMORY,
        
    )
    
    if LOAD_MODEL:
        load_checkpoint(torch.load("PFseg_checkpoint.pth.tar"), model)
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
        
        # check accuracy and plot prediction images
        check_accuracy(val_loader, model, device=DEVICE)


    # # Plot and compare images with predictions
    # for x, y in train_loader:
    #     x = x.float().unsqueeze(1).to(device=DEVICE)
    #     preds = torch.sigmoid(model(x))
    #     testing_output_label = model(preds.to(device=DEVICE))
    #     testing_output_label = testing_output_label.cpu().detach().numpy()
    #     plt.figure()
    #     plt.imshow(x[0, 0, :, :, 16], cmap='gray')
    #     plt.imshow(testing_output_label[0, 0, :, :, 16], cmap='gray', alpha=0.7)
    #     plt.show()
        
        # print some examples to a folder
        # save_predictions_as_imgs(
        #     val_loader, model, folder="/Users/kurtlab/Documents/GitHub/Brain_Segmentation/saved_BrainSegImages", device=DEVICE
        # )
    
    
    
    
if __name__ == "__main__":
    main()
