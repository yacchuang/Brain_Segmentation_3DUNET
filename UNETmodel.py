#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul  1 07:53:39 2022

@author: kurtlab
"""

import torch
import torch.nn as nn
import torchvision.transforms.functional as TF

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3,1,1, bias=False),  # 1st Conv
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3,1,1, bias=False),  # 2nd Conv
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            )
        
    def forward(self, x):
        return self.conv(x)