# Script for data augmentation functions

import numpy as np
from collections import deque
from PIL import Image
import cv2
from data.config import *

def imread_cv2(image_path):
    """
    Read image_path with cv2 format (H, W, C)
    if image is '.gif' outputs is a numpy array of {0,1}
    """
    image_format = image_path[-3:]
    if image_format == 'jpg':
        image = cv2.imread(image_path)
    else:
        image = np.array(Image.open(image_path))

    return image

def resize_cv2(image, heigh=1280, width=1918):
    return cv2.resize(image, (width, heigh), cv2.INTER_LINEAR)

def image_to_tensor(image, mean=0, std=1.):
    """Transform image (input is numpy array, read in by cv2) """
    if len(image.shape) == 2:
        image = image.reshape(image.shape[0], image.shape[1], 1)
    image = image.astype(np.float32)
    image = (image-mean)/std
    image = image.transpose((2,0,1))
    tensor = torch.from_numpy(image)

    return tensor

# --- Data Augmentation functions --- #
# A lot of functions can be found here:
# https://github.com/fchollet/keras/blob/master/keras/preprocessing/image.py#L223

# transform image and label
def randomHorizontalFlip(image, mask, p=0.5):
    """Do a random horizontal flip with probability p"""
    if np.random.random() < p:
        image = np.fliplr(image)
        mask = np.fliplr(mask)
    return image, mask

def randomVerticalFlip(image, mask, p=0.5):
    """Do a random vertical flip with probability p"""
    if np.random.random() < p:
        image = np.flipud(image)
        mask = np.flipud(mask)
    return image, mask

def randomHorizontalShift(image, mask, max_shift=0.05, p=0.5):
    """Do random horizontal shift with max proportion shift and with probability p
    Elements that roll beyond the last position are re-introduced at the first."""
    max_shift_pixels = int(max_shift*image.shape[1])
    shift = np.random.choice(np.arange(-max_shift_pixels, max_shift_pixels+1))
    if np.random.random() < p:
        image = np.roll(image, shift, axis=1)
        mask = np.roll(mask, shift, axis=1)
    return image, mask

def randomVerticalShift(image, mask, max_shift=0.05, p=0.5):
    """Do random vertical shift with max proportion shift and probability p
    Elements that roll beyond the last position are re-introduced at the first."""
    max_shift_pixels = int(max_shift*image.shape[0])
    shift = np.random.choice(np.arange(-max_shift_pixels, max_shift_pixels+1))
    if np.random.random() < p:
            image = np.roll(image, shift, axis=0)
            mask = np.roll(mask, shift, axis=0)
    return image, mask

def randomInvert(image, mask, p=0.5):
    """Randomly invert image with probability p"""
    if np.random.random() < p:
        image = 255 - image
        mask = mask
    return image, mask

def randomBrightness(image, mask, p=0.5, max_value=75):
    """With probability p, randomly increase or decrease brightness.
    See https://stackoverflow.com/questions/37822375/python-opencv-increasing-image-brightness-without-overflowing-uint8-array"""
    if np.random.random() < p:
        value = np.random.choice(np.arange(-max_value, max_value))
        print(value)
        if value > 0:
            image = np.where((255 - image) < value,255,image+value).astype(np.uint8)
        else:
            image = np.where(image < -value,0,image+value).astype(np.uint8)
    return image, mask

def GaussianBlur(image, mask, kernel=(1, 1),sigma=1,  p=0.5):
    """With probability p, apply Gaussian blur"""
    # TODO
    return image, mask

def randomRotate(image, mask, max_angle, p=0.5):
    """Perform random rotation with max_angle and probability p"""
    # TODO

    return(image, mask)
