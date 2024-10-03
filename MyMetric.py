import einops
import numpy as np
from torch import nn
import scipy
from scipy.optimize import linear_sum_assignment


class dice_coef(nn.Module):      # 这个不是loss
    def __init__(self):
        super().__init__()

    def forward(self, batch_idx, batch_size, dice, y_true, y_pred):
        smooth = 1e-6
        y_true_copy = y_true.data.cpu().numpy().astype(np.float32)
        y_pred_copy = y_pred.data.cpu().numpy().astype(np.float32)

        y_true_copy = y_true_copy / (np.max(y_true_copy))
        y_pred_copy = np.where(y_pred_copy > 0.5, 1, 0)

        y_true_copy[np.isnan(y_true_copy)] = 0.0
        y_pred_copy[np.isnan(y_pred_copy)] = 0.0

        tmp = y_pred_copy + y_true_copy
        y_pred_copy_ = einops.rearrange(y_pred_copy, "b k w h -> b (k w h)")
        y_true_copy_ = einops.rearrange(y_true_copy, "b k w h -> b (k w h)")
        tmp_ = einops.rearrange(tmp, "b k w h -> b (k w h)")
        batch, _ = tmp_.shape

        a = np.sum(np.where(tmp_ == 2, 1, 0), axis=1)
        b = np.sum(y_pred_copy_, axis=1)
        c = np.sum(y_true_copy_, axis=1)

        dice_ = ((dice * (batch_idx * batch_size) + np.sum((2 * a) / (b + c + smooth), axis=0))
                 / ((batch_idx * batch_size) + batch))

        return dice_


def get_dice_2(true, pred):
    """Ensemble Dice as used in Computational Precision Medicine Challenge.
    from FRONT BIOENG BIOTECH2019 Methods for segmentation and classification of digital microscopy tissue images
    Ensemble Dice or dice_2
    """
    true = np.copy(true)
    pred = np.copy(pred)
    true_id = list(np.unique(true))
    pred_id = list(np.unique(pred))
    # remove background aka id 0，这里是剔除了背景类
    true_id.remove(0)
    pred_id.remove(0)

    total_markup = 0
    total_intersect = 0
    for t in true_id:
        t_mask = np.array(true == t, np.uint8)
        for p in pred_id:
            p_mask = np.array(pred == p, np.uint8)
            intersect = p_mask * t_mask
            if intersect.sum() > 0:
                total_intersect += intersect.sum()
                total_markup += t_mask.sum() + p_mask.sum()
    return 2 * total_intersect / total_markup

