# Script for postprocessing functions, like making a submission file etc.

import numpy as np
from PIL import Image

def upscale_test_img(pil_img):
    """Upscale PIL Image to the shape of the test images.
    Args
    pil_img: PIL Image

    Return np array of shape (1280,1910)"""

    # Go to square of size (1280,1280) with bilinear interpolation
    im = pil_img.resize((1280,1280), resample=Image.BILINEAR)
    # Go to numpy
    im = np.array(im)/255

    # Pad with zeros to a width of 1910
    # See https://docs.scipy.org/doc/numpy/reference/generated/numpy.pad.html
    n_padding = (1910 - 1280)// 2
    im = np.pad(im,((0,0),(n_padding,n_padding)), 'constant', constant_values=(0))

    return(im)



def rle(img, threshold=0.5):
    """Fast run-lenght encoding of an image.
    See https://www.kaggle.com/hackerpoet/even-faster-run-length-encoder
    And https://stackoverflow.com/questions/3678869/pythonic-way-to-combine-two-lists-in-an-alternating-fashion
    Input
    -----
    img: image to be encoded
    threshold: cut-off probability to assign a pixel to class 1

    Returns
    -------
    start_ix: start indices
    lengths: lengths
    rle_str: resulting rle encoding as a string
    """
    flat_img = img.flatten()
    flat_img = np.where(flat_img > threshold, 1, 0).astype(np.uint8)

    starts = np.array((flat_img[:-1] == 0) & (flat_img[1:] == 1))
    ends = np.array((flat_img[:-1] == 1) & (flat_img[1:] == 0))
    starts_ix = np.where(starts)[0] + 2
    ends_ix = np.where(ends)[0] + 2
    lengths = ends_ix - starts_ix

    result = result = [None]*(len(starts_ix)+len(lengths))
    result[::2] = starts_ix
    result[1::2] = lengths
    result = [str(e) for e in result]

    rle_str = ' '.join(result)
    return starts_ix, lengths, rle_str
