# Code for UNet segmentation model architectures

import torch
import torch.nn as nn
import torch.nn.functional as F

import models.model_utils as mu # building blocks, loss functions...

# --- Architectures ---  #

class DebugNet(nn.Module):
    """Simple UNet architecture for debugging."""
    def __init__(self):
        super(DebugNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, 1, padding=0)
        self.classify = nn.Conv2d(16, 1, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.classify(x)
        x = F.sigmoid(x)
        return(x)
