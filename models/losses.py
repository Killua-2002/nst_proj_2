"""
losses.py
Implementation of Loss functions in PyTorch.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

def standard_dice_loss(y_true, y_pred_logits, smooth=1e-6):
    y_pred = torch.sigmoid(y_pred_logits)
    # y_true shape: (B, C, H, W)
    intersection = torch.sum(y_true * y_pred, dim=[1, 2, 3])
    cardinality = torch.sum(y_true + y_pred, dim=[1, 2, 3])
    dice = (2.0 * intersection + smooth) / (cardinality + smooth)
    return 1.0 - torch.mean(dice)

def standard_loss(y_true, y_pred_logits):
    bce = F.binary_cross_entropy_with_logits(y_pred_logits, y_true)
    dice = standard_dice_loss(y_true, y_pred_logits)
    return bce + dice

def fuzzy_dice_loss(y_true, y_pred_logits, smooth=1e-6):
    """
    Fuzzy Dice Loss: Giao và Hợp theo Logic Mờ (Min T-norm).
    """
    y_pred = torch.sigmoid(y_pred_logits)
    # Giao theo Logic Mờ: Min(A, B) thay vì A * B
    intersection = torch.sum(torch.min(y_true, y_pred), dim=[1, 2, 3])
    # Tổng chuẩn hóa vẫn giữ nguyên vì Min(A,B) + Max(A,B) = A + B
    cardinality = torch.sum(y_true + y_pred, dim=[1, 2, 3])
    dice = (2.0 * intersection + smooth) / (cardinality + smooth)
    return 1.0 - torch.mean(dice)

def fuzzy_bce_loss(y_true, y_pred_logits):
    """
    Fuzzy BCE Loss
    """
    return F.binary_cross_entropy_with_logits(y_pred_logits, y_true)

def combined_fuzzy_loss(y_true, y_pred_logits):
    return fuzzy_bce_loss(y_true, y_pred_logits) + fuzzy_dice_loss(y_true, y_pred_logits)
