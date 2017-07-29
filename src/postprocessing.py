# Import libraries
import numpy as np # linear algebra
from PIL import Image

def rle(img):
    """Fast run-lenght encoding of an image.
    See https://www.kaggle.com/hackerpoet/even-faster-run-length-encoder
    Input
    -----
    img: image to be encoded

    Returns
    -------
    start_ix: start indices
    lengths: lengths
    """
    flat_img = img.flatten()
    flat_img = np.where(flat_img > 0.5, 1, 0).astype(np.uint8)

    starts = np.array((flat_img[:-1] == 0) & (flat_img[1:] == 1))
    ends = np.array((flat_img[:-1] == 1) & (flat_img[1:] == 0))
    starts_ix = np.where(starts)[0] + 2
    ends_ix = np.where(ends)[0] + 2
    lengths = ends_ix - starts_ix

    return starts_ix, lengths
