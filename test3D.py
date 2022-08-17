"""
Created on WED AUG 17 2022

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
from BrainDataset import get_augmentation, patch_size
from utils import (
    load_checkpoint,
    save_checkpoint,
    get_loaders,
    check_accuracy,
    save_predictions,
)


# Hyperparameters etc.
LEARNING_RATE = 1e-4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 1  # CAN INCREASE
NUM_EPOCHS = 10
NUM_WORKERS = 0
patch_size = (128, 128, 128)  # Whole MRI image
PIN_MEMORY = True
LOAD_MODEL = True  # True

# Images directory
TEST_IMG_DIR = "/Users/kurtlab/Desktop/Brain_segmentation/BrainSeg/BrainMRI_test"


def main():
    aug = get_augmentation(patch_size)

    model = UNET(in_channels=1, out_channels=1).to(DEVICE)  # For multiclass segmentation, out_channels can increase
    loss_fn = nn.BCEWithLogitsLoss()  # For multiclass seg: cross entropy loss
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    test_loader = get_loaders(
        TEST_IMG_DIR,
        BATCH_SIZE,
        aug,
        NUM_WORKERS,
        PIN_MEMORY,
    )

    load_checkpoint(torch.load("PFcheckpoint_EP10.pth.tar"), model)
    check_accuracy(test_loader, model, device=DEVICE)

    scaler = torch.cuda.amp.GradScaler()

    # check accuracy and plot prediction images
    check_accuracy(test_loader, model, device=DEVICE)

    # print predictions to a folder
    save_predictions(
        test_loader, model, folder="/Users/kurtlab/Documents/GitHub/Brain_Segmentation/saved_BrainSegImages",
        device=DEVICE
    )

    '''
    for epoch in range(NUM_EPOCHS):
        train_fn(train_loader, model, optimizer, loss_fn, scaler)

        # save model
        checkpoint = {
            "state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict(),
        }
        save_checkpoint(checkpoint)

        # check accuracy and plot prediction images
        check_accuracy(test_loader, model, device=DEVICE)

        # print some examples to a folder
        save_predictions(
            test_loader, model, folder="/Users/kurtlab/Documents/GitHub/Brain_Segmentation/saved_BrainSegImages",
            device=DEVICE
        )
        
    '''




if __name__ == "__main__":
    main()
