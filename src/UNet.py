import os
import sys
import argparse

import pandas as pd
from tqdm import tqdm
import numpy as np
import torch
from torch import optim
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

# import custom python modules
from data.config import *
import data.data_utils as du
import models.model_utils as mu
import processing.processing_utils as pu
from models.models import UNet128 as Net

def run_script(name, epochs, batch_size, debug):
    print('Did I find a GPU? -->{}<--'.format('Yes!' if GPU_AVAIL else 'No.'))

    # Base transforms: to be applied on both input images and output masks
    base_tsfm = transforms.Compose([transforms.Scale(128),
                                    transforms.CenterCrop(128),
                                    transforms.ToTensor()])

    # Datasets
    train_dataset = du.CarvanaDataset('../data/raw/train', '../data/raw/train_masks/',
                                      common_transforms=base_tsfm, debug=debug)
    test_dataset = du.CarvanaDataset('../data/raw/test',
                                     input_transforms=base_tsfm, debug=debug)

    # DataLoaders
    train_loader = DataLoader(train_dataset, shuffle=True, batch_size=batch_size, num_workers=3)
    test_loader = DataLoader(test_dataset, shuffle=False, batch_size=batch_size, num_workers=3)

    # Model
    if GPU_AVAIL:
        net = Net().cuda()
    else:
        net = Net()

    # Optimizer
    optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.9, weight_decay=0.0005)

    # Loss
    def criterion(logits, labels):
        """Define loss function to be used. Can be a mix of losses defined in model_utils module."""
        l = mu.BCELoss2d()(logits, labels) + mu.DiceLoss()(logits, labels)
        return l

    # Training loop
    loss_history = []
    print('Training for {} epochs...'.format(epochs))
    for epoch in tqdm(range(epochs)):

        for i, im in enumerate(train_loader):
            if GPU_AVAIL:
                images = Variable(im['image'].cuda())
                masks_gt = Variable(im['mask'].cuda())
            else:
                images = Variable(im['image'])
                masks_gt = Variable(im['mask'])

            # forward pass
            masks_pred = net.train()(images)

            # backward pass
            loss = criterion(masks_pred, masks_gt)
            optimizer.zero_grad() # set all gradients to zero
            loss.backward() # backpropagate
            optimizer.step() # do update step

            if i % 1 == 0:
                loss_history.append(loss.data.cpu().numpy()[0])

    print('Making predictions for test data...')

    test_idx = []
    rle_encoded_predictions = []

    for i, im in enumerate(tqdm(test_loader)):
        if GPU_AVAIL:
            images = Variable(im['image'].cuda())
        else:
            images = images = Variable(im['image'])

        masks_test = net.eval()(images)

        # Go from pytorch tensor to list of PIL images, which can be rescaled and interpolated
        PIL_list = [transforms.ToPILImage()(masks_test.data[b].cpu()) for b in range(masks_test.size()[0])]

        # Rescale them to np matrices with the correct size
        np_list = [pu.upscale_test_img(img) for img in PIL_list]

        # rle encode the predictions
        rle_encoded_predictions.append([pu.rle(im)[2] for im in np_list])
        test_idx.append(im['id'])


    # Prepare submission file
    test_idx_all = [j+'.jpg' for batch in test_idx for j in batch]
    rle_encoded_predictions_all = [j for batch in rle_encoded_predictions for j in batch]
    predictions_mapping = dict(zip(test_idx_all, rle_encoded_predictions_all))

    # Map predictions to the sample submission file to make sure we make no errors with the ordering of files
    submission_file = pd.read_csv(SAMPLE_SUB_CSV)
    submission_file['rle_mask'] = submission_file['img']
    submission_file['rle_mask'] = submission_file['rle_mask'].map(predictions_mapping)

    submission_file.to_csv('../predictions/test/{}.csv'.format(name), index=False)
    print('Done!')


def main():
    parser = argparse.ArgumentParser(description='UNet segmentation network for Kaggle Carvana competition.')
    parser.add_argument('name', type=str, help='Session name.')
    parser.add_argument('epochs', type=int, help='Number of training epochs')
    parser.add_argument('-b','--batch_size', type=int, default=32, help='(optional) Batch size')
    parser.add_argument('-db', '--debug', action='store_true', help='(optional) Debug mode')
    args = parser.parse_args()

    run_script(**vars(args))

if __name__ == "__main__":
    sys.exit(main())
