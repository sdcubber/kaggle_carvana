from os.path import join as join_path
import numpy as np
import matplotlib.pyplot as plt
import torch
import random
import sys
import shutil
import os
from datetime import datetime
import argparse
import pandas as pd
from torch.autograd import Variable
import torch
import torch.nn as nn
import torch.nn.functional as F

INPUT_PATH = '../data/'
OUTPUT_SUB_PATH = '../predictions/'
OUTPUT_WEIGHT_PATH = '../models/'
OUTPUT_TEMP_PATH = '../temp/'

TRAIN_IMG_PATH = join_path(INPUT_PATH,'raw/train/')
TEST_IMG_PATH = join_path(INPUT_PATH,'raw/test/')
TRAIN_MASKS_PATH = join_path(INPUT_PATH,'raw/train_masks')

TRAIN_MASKS_CSV = join_path(INPUT_PATH, 'train_masks.csv')
SAMPLE_SUB_CSV = join_path(INPUT_PATH, 'sample_submission.csv')
META_DATA_CSV = join_path(INPUT_PATH, 'metadata.csv')

# Check if there is a GPU available
GPU_AVAIL = torch.cuda.is_available()
