# -*- coding: utf-8 -*-

import torch
import einops
import numpy as np
from torch import nn

class cross_loss(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, y_true, y_pred):
        smooth = 1e-6
        return -torch.mean(y_true * torch.log(y_pred + smooth) +
                           (1 - y_true) * torch.log(1 - y_pred + smooth))


