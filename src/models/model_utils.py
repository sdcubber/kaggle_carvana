# Model utility functions such as loss functions, CNN building blocks etc.

import torch
import torch.nn as nn
import torch.nn.functional as F

# --- Custom loss functions --- #

class DiceLoss(nn.Module):

    def __init__(self, weight=None, size_average=True):
        super(DiceLoss, self).__init__()

    def forward(self, logits, targets):
        num = targets.size(0)
        probs = F.sigmoid(logits)
        m1  = probs.view(num,-1)
        m2  = targets.view(num,-1)
        intersection = (m1 * m2)

        score = 2. * (intersection.sum(1)+1) / (m1.sum(1) + m2.sum(1)+1)
        score = 1- score.sum()/num
        return score


# --- CNN building blocks --- #
