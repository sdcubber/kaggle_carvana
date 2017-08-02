# Script for postprocessing functions, like making a submission file etc.

import numpy as np
from PIL import Image

def upscale_test_img(pil_img):
    """Upscale PIL Image to the shape of the test images.
    Args
    pil_img: PIL Image

    Return np array of shape (1280,1918)"""

    # Go to square of size (1280,1280) with bilinear interpolation
    im = pil_img.resize((1280,1280), resample=Image.BILINEAR)
    # Go to numpy
    im = np.array(im)/255

    # Pad with zeros to a width of 1918
    # See https://docs.scipy.org/doc/numpy/reference/generated/numpy.pad.html
    n_padding = (1918 - 1280)// 2
    im = np.pad(im,((0,0),(n_padding,n_padding)), 'constant', constant_values=(0))

    return(im)

def rle(img):
    """
    img: numpy array, 1 - mask, 0 - background
    Returns run length as string formated
    """
    pixels = img.flatten()
    pixels[0] = 0
    pixels[-1] = 0
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 2
    runs[1::2] = runs[1::2] - runs[:-1:2]

    return ' '.join(str(x) for x in runs)

def rle_decode(im_rle, shape=(1280,1918)):
    """
    im_rle: run-length encoded image
    shape: (height,width) of array to return
    Returns numpy array, 1 - mask, 0 - background
    """

    s = im_rle.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths
    img = np.zeros(shape[0]*shape[1], dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1

    return img.reshape(shape)
