import os
from torch.utils.data import Dataset
from PIL import Image
from data.config import*

class CarvanaDataset(Dataset):
    """Kaggle Carvana dataset."""

    def __init__(self, im_dir, mask_dir=None, common_transforms=None, input_transforms=None, rotation_ids=range(1,17), debug=False):
        """
        Args:
        im_dir (string): Directory with all the images.
        mask_dir (string): Directory with the masks. None for test data.
        common_transforms: transforms on input and output images
        input_transforms: transforms only on input images
        rotation_ids (list [1-16]): containing types of rotations that we are interested
        debug (boolean): In debug mode, use only 100 images
        """

        self.im_dir = im_dir # directory
        self.mask_dir = mask_dir
        self.im_list = [name.split('.')[0] for name in os.listdir(im_dir)]

        # get all files with specific rotations in rotation_ids
        rotation_ids = np.char.mod('_%02d', rotation_ids)
        self.im_list = [item for item in self.im_list if any(rot_id in item for rot_id in rotation_ids)]

        if debug:
            self.im_list = self.im_list[:64]

        self.common_transforms = common_transforms
        self.input_transforms = input_transforms

    def __len__(self):
        return len(self.im_list)

    def __getitem__(self, idx):
        im_name = self.im_list[idx]
        image = Image.open(join_path(self.im_dir, im_name + '.jpg'))
        mask = None

        if self.mask_dir:
            mask = Image.open(join_path(self.mask_dir, im_name + '_mask.gif'))

        if self.input_transforms:
            image=self.input_transforms(image)

        if self.common_transforms:
            image=self.common_transforms(image)
            mask=self.common_transforms(mask)
        if self.mask_dir:
            return {'image':image, 'mask':mask, 'id': im_name}
        else:
            return {'image':image, 'id': im_name}
