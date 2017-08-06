import os
from torch.utils.data import Dataset
from PIL import Image
from data.config import *


class CarvanaDataset(Dataset):
    """Kaggle Carvana dataset."""

    def __init__(self, im_dir,
                 ids_list=None,
                 mask_dir=None,
                 input_transforms=None,
                 mask_transforms=None,
                 rotation_ids=range(1, 17),
                 debug=False):
        """
        Args:
        im_dir (string): Directory with all the images.
        ids_list (string array): list of all file ids (train or valid)
        mask_dir (string): Directory with the masks. None for test data.
        input_transforms: transforms only on input images
        mask_transforms: transforms on input and output images
        rotation_ids (list [1-16]): containing types of rotations that we are interested
        debug (boolean): In debug mode, use only 100 images
        """
        self.im_dir = im_dir
        self.mask_dir = mask_dir

        if ids_list is None:
            ids_list = os.listdir(im_dir)

        self.im_list = [name.split('.')[0] for name in ids_list]
        # get all files with specific rotations in rotation_ids
        rotation_ids = np.char.mod('_%02d', rotation_ids)
        self.im_list = [item for item in self.im_list
                        if any(rot_id in item for rot_id in rotation_ids)]
        if debug:
            self.im_list = self.im_list[:64]

        self.input_transforms = input_transforms
        self.mask_transforms = mask_transforms

    def __getitem__(self, idx):
        im_name = self.im_list[idx]
        image = Image.open(os.path.join(self.im_dir, im_name + '.jpg'))
        mask = 0

        if self.input_transforms:
            image = self.input_transforms(image)

        if self.mask_dir:
            mask = Image.open(os.path.join(self.mask_dir, im_name + '_mask.gif'))  # .convert(mode='L')

            if self.mask_transforms:
                mask = self.mask_transforms(mask)

            mask = np.array(mask, np.float32).reshape(1, mask.size[0], mask.size[1])
            mask = torch.from_numpy(mask)

        return image, mask, im_name #it should be faster than a map

    def __len__(self):
        return len(self.im_list)
