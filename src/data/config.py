from os.path import join as join_path
import numpy as np

INPUT_PATH = '../data/'
OUTPUT_SUB_PATH = '../predictions/'
OUTPUT_WEIGHT_PATH = '../models/'

TRAIN_IMG_PATH = join_path(INPUT_PATH,'raw/train/')
TEST_IMG_PATH = join_path(INPUT_PATH,'raw/test/')
TRAIN_MASKS_PATH = join_path(INPUT_PATH,'raw/train_masks')

TRAIN_MASKS_CSV = join_path(INPUT_PATH, 'train_masks.csv')
SAMPLE_SUB_CSV = join_path(INPUT_PATH, 'sample_submission.csv')
META_DATA_CSV = join_path(INPUT_PATH, 'metadata.csv')