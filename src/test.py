
from data.data_utils import CarvanaDataset
from models.models import UNet128
from models.model_utils import *
from processing.processing_utils import *
from torch.utils.data import DataLoader

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
"""
car_code = "00087a6bd4dc"
angle_code = "04"

print(read_mask_image(car_code, angle_code))
show_mask_image(car_code, angle_code)

gp = pd.read_csv(TRAIN_MASKS_CSV)
"""

a = torch.from_numpy(np.array([[1, 1], [0.1, 0.3]]).astype(np.float32))
b = torch.from_numpy(np.array([[0., 1], [2, 0.9]]).astype(np.float32))

a = Variable(a)
b = Variable(b)
print(a.sum())
print(a*b)

criterion = nn.BCELoss()

print(criterion(a, b))