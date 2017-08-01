# Script with utility code for data

import torch
import os
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import ToTensor
from PIL import Image

pil_to_tensor = ToTensor()

class CarvanaDataset(Dataset):
    """Kaggle Carvana dataset."""

    def __init__(self, im_dir, mask_dir=None,
     common_transforms=None,
     input_transforms=None,
     debug=False):
        """
        Args:
        im_dir (string): Directory with all the images.
        mask_dir (string): Directory with the masks. None for test data.
        common_transforms: transforms on input and output images
        input_transforms: transforms only on input images
        debug (boolean): In debug mode, use only 100 images
        """

        self.im_dir = im_dir # directory
        self.im_list = os.listdir(self.im_dir) # list with image names
        if debug:
            self.im_list = self.im_list[:100]

        self.common_transforms = common_transforms
        self.input_transforms = input_transforms

        if mask_dir:
            self.mask_dir = mask_dir
            self.mask_list = os.listdir(self.mask_dir)
            # list with id's to match them with training images
            self.maskid_list = [name.split('.')[0].split('_mask')[0] for name in self.mask_list]

    def __len__(self):
        return(len(self.im_list))

    def __getitem__(self, idx):
        im_name = os.path.join(self.im_dir, self.im_list[idx])
        im_id = self.im_list[idx].split('.')[0]

        image = Image.open(im_name)
        mask = None

        if self.mask_dir:
            mask_loc = self.maskid_list.index(im_id)
            mask_name = os.path.join(self.mask_dir, self.mask_list[mask_loc])
            mask = Image.open(mask_name)

        if self.input_transforms:
       	    image=self.input_transforms(image)

        if self.common_transforms:
            image=self.common_transforms(image)
            mask=self.common_transforms(mask)

        sample={'image':image, 'mask':mask}
        return(sample)
