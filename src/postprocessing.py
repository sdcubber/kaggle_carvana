# Script for postprocessing functions, like making a submission file etc.

import numpy as np
from PIL import Image

def rle(img):
    """Fast run-lenght encoding of an image.
    See https://www.kaggle.com/hackerpoet/even-faster-run-length-encoder
    And https://stackoverflow.com/questions/3678869/pythonic-way-to-combine-two-lists-in-an-alternating-fashion
    Input
    -----
    img: image to be encoded

    Returns
    -------
    start_ix: start indices
    lengths: lengths
    rle_str: resulting rle encoding as a string
    """
    flat_img = img.flatten()
    flat_img = np.where(flat_img > 0.5, 1, 0).astype(np.uint8)

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
