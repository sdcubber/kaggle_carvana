# Script for postprocessing functions, like making a submission file etc.

import numpy as np
from collections import deque
from PIL import Image
from data.config import *


def make_prediction_file(output_file, sample_file, test_ids, rle_encoded_preds, train_data=False):
    """
    Create a prediction file
    Args:
    output_file: the name of output file
    test_ids: ids of test files
    rle_encoded_preds: encoded strings
    train_data: if True, make submission file for the training data instead of test data
    """

    # Prepare submission file
    predictions_mapping = dict(zip([j + '.jpg' for j in test_ids], rle_encoded_preds))

    # Map predictions to the sample submission file to make sure we make no errors with the ordering of files
    submission_file = pd.read_csv(sample_file)
    submission_file['rle_mask'] = submission_file['img'].map(predictions_mapping)
    submission_file.to_csv(output_file, index=False, compression='gzip')


def upscale_test_img(pil_img, crop=False):
    """Upscale PIL Image to the shape of the test images.
    Args
    pil_img: PIL Image
    crop: True if original images were centered cropped (requires different upscaling)
    Return np array of shape (1280,1918)"""

    if crop:
        # Go to square of size (1280,1280) with bilinear interpolation
        im = pil_img.resize((1280, 1280), resample=Image.BILINEAR)
        # Go to numpy
        im = np.array(im) // 255
        # Pad with zeros to a width of 1918
        # See https://docs.scipy.org/doc/numpy/reference/generated/numpy.pad.html
        n_padding = (1918 - 1280) // 2
        im = np.pad(im, ((0, 0), (n_padding, n_padding)), 'constant', constant_values=(0))

    else:
        # Go to original resolution by upscaling, no padding required since no cropping was done
        im = pil_img.resize((1918, 1280), resample=Image.ANTIALIAS)
        im = np.array(im) // 255

    return (im)


def compute_weight(mask, wc=1, wo=2, sigma=10):
    """
    Compute weights for all pixels
    mask: a numpy array of pixels
    Return a numpy weight array for each pixel in img_mask 
    """
    img_mask = mask.tolist()
    moves = [(-1, 0), (1, 0), (0, 1), (0, -1)]
    (w, h) = mask.shape
    Q = deque()
    weights = [[-1 for j in range(h)] for i in range(w)]

    # finding pixels in border
    for i in range(w):
        for j in range(h):
            is_border = False
            for k in range(4):
                x, y = i + moves[k][0], j + moves[k][1]
                if 0 <= x < w and 0 <= y < h:
                    is_border |= img_mask[i][j] != img_mask[x][y]
            if is_border:
                Q.append((i, j, 0))
                weights[i][j] = wc + wo

    # compute distance
    while Q:
        (i, j, d) = Q.popleft()
        for k in range(4):
            x, y = i + moves[k][0], j + moves[k][1]
            if 0 <= x < w and 0 <= y < h and weights[x][y] < 0:
                weights[x][y] = wc + wo * math.exp(-((d + 1) / sigma) ** 2)
                Q.append((x, y, d + 1))

    return np.array(weights)


def rle_encode(mask_image):
    """
    img: numpy array, 1 - mask, 0 - background
    Returns run length as string formated
    """
    pixels = np.append(np.append(0, mask_image.flatten()), 0)
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] = runs[1::2] - runs[:-1:2]

    return ' '.join(str(x) for x in runs)


def rle_decode(im_rle, shape=(1280, 1918)):
    """
    im_rle: run-length encoded image
    shape: (height,width) of array to return
    Returns numpy array, 1 - mask, 0 - background
    """

    s = im_rle.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths
    img = np.zeros(shape[0] * shape[1], dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1

    return img.reshape(shape)


def read_mask_image(car_code, angle_code):
    """
    Read image mask, encoding to 0-black 1-white
    car_code: code of the car
    angle_code: code of the angle
    """
    mask_img_path = os.path.join(TRAIN_MASKS_PATH, car_code + '_' + angle_code + '_mask.gif')
    mask_img = np.array(Image.open(mask_img_path))  # .convert(mode='L'))

    return mask_img


def show_mask_image(car_code, angle_code):
    """
    Show the image mask
    """
    mask_img = read_mask_image(car_code, angle_code)
    plt.imshow(mask_img, cmap='Greys_r')
    plt.show()


def train_valid_split(csvfile, rotation_ids=range(1, 17), valid=0.1):
    """ Return a list of ids for training and a list for validation"""
    im_list = pd.read_csv(csvfile)['img']
    im_list = [item.split('_')[0] for item in im_list]
    im_list = np.array(np.unique(im_list))

    np.random.shuffle(im_list)

    t_size = int(im_list.shape[0] * (1 - valid))
    train, valid = im_list[:t_size], im_list[t_size:]

    train = [item + '_{:02d}'.format(rot_id) for rot_id in rotation_ids for item in train]
    valid = [item + '_{:02d}'.format(rot_id) for rot_id in rotation_ids for item in valid]

    return np.array(train), np.array(valid)


def update_spreadsheet(timestamp, im_size, arch, epochs, best_dice, best_loss, modelname, rotation):
    """Update the spreadsheet with information from this experiment"""
    # TODO

    return (True)


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class Logger(object):
    """ Write log info to file and to standard output  """

    def __init__(self):
        self.terminal = sys.stdout  # stdout
        self.file = None

    def open(self, file, mode=None):
        if mode is None: mode = 'w'
        self.file = open(file, mode)

    def write(self, message: object, is_terminal: object = 1, is_file: object = 1) -> object:
        if '\r' in message: is_file = 0

        if is_terminal == 1:
            self.terminal.write(message)
            self.terminal.flush()

        if is_file == 1:
            self.file.write(message)
            self.file.flush()

    def flush(self):
        # this flush method is needed for python 3 compatibility.
        # this handles the flush command by doing nothing.
        # you might want to specify some extra behavior here.
        pass
