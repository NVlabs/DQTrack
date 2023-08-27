# Copyright (c) 2023, NVIDIA Corporation & Affiliates. All rights reserved. 
# 
# This work is made available under the Nvidia Source Code License-NC. 
# To view a copy of this license, visit 
# https://github.com/NVlabs/DQTrack/blob/main/LICENSE
import torch
import torch.nn as nn
import torch.nn.functional as F

from mmdet.models.builder import LOSSES
from mmdet.models.losses.utils import weight_reduce_loss

@LOSSES.register_module()
class CategoricalCrossEntropyLoss(nn.Module):
    def __init__(self,
                 reduction='mean',
                 class_weight=None,
                 loss_weight=1.0):
        """CrossEntropyLoss.
        Args:
            reduction (str, optional): . Defaults to 'mean'.
                Options are "none", "mean" and "sum".
            class_weight (list[float], optional): Weight of each class.
                Defaults to None.
            loss_weight (float, optional): Weight of the loss. Defaults to 1.0.
        """
        super(CategoricalCrossEntropyLoss, self).__init__()
        self.reduction = reduction
        self.loss_weight = loss_weight
        self.class_weight = class_weight


    def forward(self,
                cls_score,
                label_score,
                weight=None,
                avg_factor=None,
                reduction_override=None,
                eps=1e-9,
                **kwargs):
        """Forward function.
        Args:
            cls_score (torch.Tensor): The prediction.
            label_score (torch.Tensor): The learning label score of the prediction.
            weight (torch.Tensor, optional): Sample-wise loss weight.
            avg_factor (int, optional): Average factor that is used to average
                the loss. Defaults to None.
            reduction (str, optional): The method used to reduce the loss.
                Options are "none", "mean" and "sum".
        Returns:
            torch.Tensor: The calculated loss
        """
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)
        
        if self.class_weight is not None:
            class_weight = cls_score.new_tensor(
                self.class_weight, device=cls_score.device)
        else:
            class_weight = None
        
        # element-wise losses
        cls_score = torch.clamp(cls_score, eps, 1 - eps)
        loss = - (label_score * cls_score.log())
        if class_weight is not None:
            loss = loss * class_weight
        loss = loss.sum(dim=-1)
        # apply weights and do the reduction
        if weight is not None:
            weight = weight.float()
        loss = weight_reduce_loss(
            loss, weight=weight, reduction=reduction, avg_factor=avg_factor)
        
        loss_cls = self.loss_weight * loss
        return loss_cls