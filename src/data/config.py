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
OUTPUT_LOG_PATH = '../logs/'

TRAIN_IMG_PATH = os.path.join(INPUT_PATH,'raw/train/')
TEST_IMG_PATH = os.path.join(INPUT_PATH,'raw/test/')
TRAIN_MASKS_PATH = os.path.join(INPUT_PATH,'raw/train_masks')

TRAIN_MASKS_CSV = os.path.join(INPUT_PATH, 'train_masks.csv')
SAMPLE_SUB_CSV = os.path.join(INPUT_PATH, 'sample_submission.csv')
META_DATA_CSV = os.path.join(INPUT_PATH, 'metadata.csv')

# Check if there is a GPU available
GPU_AVAIL = torch.cuda.is_available()

#threshold used for prediction
THRED = 0.5