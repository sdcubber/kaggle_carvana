# Script to generate the crops

import numpy as np
import torch
from torch import optim
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import PIL.Image as Image

# import custom python modules
from data.config import *
import data.data_utils as du
import models.model_utils as mu
import processing.processing_utils as pu
import processing.augmentation as pa
from models.models import UNet128 as Net
import cv2

from tqdm import tqdm
from PIL import Image


# helpers

def vertical_split(image):
    """Perform vertical split in four images of size 512, with overlap to account for custom image size."""
    n_overlap = (2048 - image.shape[1]) // 2 # 65 pixels overlap each time

    im1 = image[:,0:512,:]
    im2 = image[:,(512-n_overlap):(1024-n_overlap),:]
    im3 = image[:,(1024-n_overlap):(1536-n_overlap),:]
    im4 = image[:,-512:,:]

    return [im1, im2, im3, im4]

def horizontal_split(image):
    n_overlap = (512*3 - image.shape[0]) // 2

    im1 = image[0:512,:,:]
    im2 = image[(512-n_overlap):(1024-n_overlap),:]
    im3 = image[(1024-2*n_overlap)::,:,:]

    return[im1, im2, im3]

def split_im(image):
    if image.ndim == 2:
        image = np.expand_dims(image, axis=2)
    """Split image into 8 smaller sub-images"""
    skip = image.shape[0] % 512 // 2 # skip top and bottom 128 pixels
    horizontal_ims = horizontal_split(image)

    sub_ims = []
    for panel in horizontal_ims:
        sub_ims.extend(vertical_split(panel))

    return sub_ims

def reconstruct_im(sub_images):
    """reconstruct original image from 12 sub images"""
    n_overlap = (2048 - image.shape[1]) // 2

    top = np.concatenate((split[0], split[1][:,n_overlap:], split[2][:,:], split[3][:,n_overlap:]), axis=1)
    middle = np.concatenate((split[4], split[5][:,n_overlap:], split[6][:,:], split[7][:,n_overlap:]), axis=1)
    bottom = np.concatenate((split[8], split[9][:,n_overlap:], split[10][:,:], split[11][:,n_overlap:]), axis=1)

    n_overlap = (512*3 - image.shape[0]) // 2

    im = np.concatenate((top, middle[n_overlap:-n_overlap,:,:], bottom), axis=0)

    return(im)

def main():
    im_list_test = os.listdir('../data/raw/test/')
    im_list_train = os.listdir('../data/raw/train/')
    mask_list = os.listdir('../data/raw/train_masks/')

    for m in tqdm(mask_list)
        mask = pa.imread_cv2(os.path.join('../data/raw/test/', m))
        split = split_im(mask)
        for i, s in enumerate(split):
            # for the images
            #cv2.imwrite(os.path.join('../data/raw/test_cropped/', m.split('.')[0] + '_{}.jpg'.format(i)),s)
            # for the masks
            im = Image.fromarray(np.squeeze(s)*255)
            im.save(os.path.join('../data/raw/train_masks_cropped/', m.split('.')[0] + '_{}.gif'.format(i)))

if __name__ == '__main__':
    # random.seed(123456789) # Fix seed
    sys.exit(main())
