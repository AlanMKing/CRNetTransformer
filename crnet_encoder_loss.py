import torch
import torch.nn as nn
import torch.nn.functional as F
from config import *


class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-6):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, pred, target):
        # 展平 pred 和 target
        pred = pred.contiguous().view(-1)
        target = target.contiguous().view(-1)

        # 计算分子，即预测和目标之间的交集
        intersection = (pred * target).sum()

        # 计算分母，即预测和目标的平方和
        denominator = pred.pow(2).sum() + target.pow(2).sum()

        # 计算 Dice Loss
        dice_loss = 1 - (2 * intersection + self.smooth) / (denominator + self.smooth)

        return dice_loss


# CRNet损失函数
class CRNETENCODERLoss(nn.Module):
    def __init__(self, lambda_full=2, lambda_center=1, lambda_reg=0.05):
        super(CRNETENCODERLoss, self).__init__()
        self.lambda_full = lambda_full
        self.lambda_center = lambda_center
        self.lambda_reg = lambda_reg
        # self.bce_loss = nn.BCEWithLogitsLoss()  # 使用BCEWithLogitsLoss而不是Dice loss
        self.bce_loss = DiceLoss()

    def forward(self, out, gts):

        loss_full_all = self.bce_loss(out['tf_all'].squeeze(1), gts['Tf'].to('cuda'))
        loss_center_all = self.bce_loss(out['tc_all'].squeeze(1), gts['Tc'].to(device))
        loss_offset_x_all = F.smooth_l1_loss(out['xf_all'].squeeze(1), gts['Xoffset'].to(device))
        loss_offset_y_all= F.smooth_l1_loss(out['yf_all'].squeeze(1), gts['Yoffset'].to(device))
        loss_reg_all = loss_offset_x_all + loss_offset_y_all

        total_loss_all = self.lambda_full * loss_full_all + self.lambda_center * loss_center_all + self.lambda_reg * loss_reg_all

        loss_full_crnet = self.bce_loss(out['tf_crnet'].squeeze(1), gts['Tf'].to('cuda'))
        loss_center_crnet = self.bce_loss(out['tc_crnet'].squeeze(1), gts['Tc'].to(device))
        loss_offset_x_crnet = F.smooth_l1_loss(out['xf_crnet'].squeeze(1), gts['Xoffset'].to(device))
        loss_offset_y_crnet = F.smooth_l1_loss(out['yf_crnet'].squeeze(1), gts['Yoffset'].to(device))
        loss_reg_crnet = loss_offset_x_crnet + loss_offset_y_crnet

        total_loss_crnet = self.lambda_full * loss_full_crnet + self.lambda_center * loss_center_crnet + self.lambda_reg * loss_reg_crnet

        loss_dict = {'loss_all': total_loss_all, 'loss_crnet': total_loss_crnet}

        return loss_dict