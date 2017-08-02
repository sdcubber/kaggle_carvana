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


def make_conv_bn_relu(in_channels, out_channels, kernel_size=3, stride=1, padding=1):
    return [
        nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
    ]


class UNet128 (nn.Module):

    def __init__(self):
        super(UNet128, self).__init__()

        #128
        self.down1 = nn.Sequential(
            *make_conv_bn_relu(3, 16, kernel_size=3, stride=1, padding=1),
            *make_conv_bn_relu(16, 32, kernel_size=1, stride=1, padding=0),
        )
        #64

        self.down2 = nn.Sequential(
            *make_conv_bn_relu(32, 64,  kernel_size=3, stride=1, padding=1),
            *make_conv_bn_relu(64, 128, kernel_size=1, stride=1, padding=0),
        )
        #32

        self.down3 = nn.Sequential(
            *make_conv_bn_relu(128, 256, kernel_size=3, stride=1, padding=1),
            *make_conv_bn_relu(256, 512, kernel_size=1, stride=1, padding=0),
        )
        #16

        self.down4 = nn.Sequential(
            *make_conv_bn_relu(512,512, kernel_size=3, stride=1, padding=1),
            *make_conv_bn_relu(512,512, kernel_size=1, stride=1, padding=0),
        )
        #8

        self.same = nn.Sequential(
            *make_conv_bn_relu(512,512, kernel_size=3, stride=1, padding=1),
        )

        #16
        self.up4 = nn.Sequential(
            *make_conv_bn_relu(1024,512, kernel_size=1, stride=1, padding=0),
            *make_conv_bn_relu( 512,512, kernel_size=3, stride=1, padding=1),
            #nn.Dropout(p=0.10),
        )
        #16

        self.up3 = nn.Sequential(
            *make_conv_bn_relu(1024,512, kernel_size=1, stride=1, padding=0),
            *make_conv_bn_relu( 512,128, kernel_size=3, stride=1, padding=1),
        )
        #32

        self.up2 = nn.Sequential(
            *make_conv_bn_relu(256,128, kernel_size=1, stride=1, padding=0),
            *make_conv_bn_relu(128, 32, kernel_size=3, stride=1, padding=1),
        )
        #64

        self.up1 = nn.Sequential(
            *make_conv_bn_relu(64, 64, kernel_size=1, stride=1, padding=0),
            *make_conv_bn_relu(64, 32, kernel_size=3, stride=1, padding=1),
        )
        #128

        self.classify = nn.Conv2d(32, 1, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        #128
        down1 = self.down1(x)
        out   = F.max_pool2d(down1, kernel_size=2, stride=2) #64

        down2 = self.down2(out)
        out   = F.max_pool2d(down2, kernel_size=2, stride=2) #32

        down3 = self.down3(out)
        out   = F.max_pool2d(down3, kernel_size=2, stride=2) #16

        down4 = self.down4(out)
        out   = F.max_pool2d(down4, kernel_size=2, stride=2) # 8

        out   = self.same(out)

        out   = F.upsample_bilinear(out, scale_factor=2) #16
        out   = torch.cat([down4, out],1)
        out   = self.up4(out)

        out   = F.upsample_bilinear(out, scale_factor=2) #32
        out   = torch.cat([down3, out],1)
        out   = self.up3(out)

        out   = F.upsample_bilinear(out, scale_factor=2) #64
        out   = torch.cat([down2, out],1)
        out   = self.up2(out)

        out   = F.upsample_bilinear(out, scale_factor=2) #128
        out   = torch.cat([down1, out],1)
        out   = self.up1(out)
        out   = self.classify(out)
        out   = F.sigmoid(out)

        return out
