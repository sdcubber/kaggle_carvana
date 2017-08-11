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
        image = image.reshape(image.shape[0], image.shape[1], 1)

    return image


def image_to_tensor(image, mean=0, std=1.):
    """Transform image (input is numpy array, read in by cv2) """
    image = image.astype(np.float32)
    image = (image-mean)/std
    image = image.transpose((2,0,1))
    tensor = torch.from_numpy(image)

    return tensor