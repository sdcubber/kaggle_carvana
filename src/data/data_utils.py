import os
from torch.utils.data import Dataset
from PIL import Image
import processing.processing_utils as pu
from data.config import *
import processing.augmentation as pa


class CarvanaDataset(Dataset):
    """Kaggle Carvana dataset."""

    def __init__(self, im_dir,
                 ids_list=None,
                 mask_dir=None,
                 common_transforms=None,
                 input_transforms=None,
                 mask_transforms=None,
                 rotation_ids=range(1, 17),
                 weighted=False,
                 debug=False):
        """
        Args:
        im_dir (string): Directory with all the images.
        ids_list (string array): list of all file ids (train or valid)
        mask_dir (string): Directory with the masks. None for test data.
        im_size = Image size. Scaling is done in PIL.
        common_transforms: transforms for both input and output images
        input_transforms: transforms only on input images
        mask_transforms: transforms on input and output images
        rotation_ids (list [1-16]): containing types of rotations that we are interested
        weighted: if we used weights for each pixel
        debug (boolean): In debug mode, use only 100 images
        """
        self.im_dir = im_dir
        self.mask_dir = mask_dir
        self.weighted = weighted

        if ids_list is None:
            ids_list = os.listdir(im_dir)

        self.im_list = [name.split('.')[0] for name in ids_list]
        # get all files with specific rotations in rotation_ids
        rotation_ids = np.char.mod('_%02d', rotation_ids)
        self.im_list = [item for item in self.im_list
                        if any(rot_id in item for rot_id in rotation_ids)]
        if debug:
            self.im_list = self.im_list[:64]

        self.common_transforms = common_transforms
        self.input_transforms = input_transforms
        self.mask_transforms = mask_transforms

    def __getitem__(self, idx):
        mask, weight = 0, 0
        im_name = self.im_list[idx]

        image = pa.imread_cv2(os.path.join(self.im_dir, im_name + '.jpg'))

        if self.mask_dir:
            mask = pa.imread_cv2(os.path.join(self.mask_dir, im_name + '_mask.gif'))

            # applying common transforms
            if self.common_transforms:
                for trans in self.common_transforms:
                    image, mask = trans(image, mask)

            # applying mask transforms
            if self.mask_transforms:
                mask = self.mask_transforms(mask)

            if self.weighted:
                weight = pu.compute_weight(mask.astype(np.int32))
                weight = weight.reshape(1, weight.shape[0], weight.shape[1])
                weight = torch.from_numpy(weight.astype(np.float32)) # convert to tensor

            # convert to tensor
            mask = pa.image_to_tensor(mask)

        # applying input transforms
        if self.input_transforms:
            image = self.input_transforms(image)
        # convert to tensor
        image = pa.image_to_tensor(image/255)

        return image, mask, weight, im_name

    def __len__(self):
        return len(self.im_list)
