# Copyright (c) 2023, NVIDIA Corporation & Affiliates. All rights reserved. 
# 
# This work is made available under the Nvidia Source Code License-NC. 
# To view a copy of this license, visit 
# https://github.com/NVlabs/DQTrack/blob/main/LICENSE
# Modified from MOTR (https://github.com/megvii-model/MOTR/blob/main/models/motr.py)
# ------------------------------------------------------------------------
# Copyright (c) 2021 megvii-model. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from Deformable DETR (https://github.com/fundamentalvision/Deformable-DETR)
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------
import torch
import torch.nn.functional as F
import torch.nn as nn
from typing import List

from mmdet.core import build_assigner, build_sampler, multi_apply, reduce_mean
from mmcv.runner import force_fp32, auto_fp16
from mmdet.models import build_loss
from mmdet.models.builder import LOSSES

from projects.mmdet3d_plugin.core.track.instances import Instances
from projects.mmdet3d_plugin.core.bbox.util import normalize_bbox

@LOSSES.register_module()
class DualQueryMatcher(nn.Module):
    def __init__(self, 
                 num_classes,
                 class_dict=None,
                 code_weights=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.2, 0.2, 0.2], 
                 sync_cls_avg_factor=False,
                 bg_cls_weight=0,
                 assigner=dict(
                     type='HungarianAssigner3D',
                     cls_cost=dict(type='FocalLossCost', weight=2.0),
                     reg_cost=dict(type='BBox3DL1Cost', weight=0.25),
                     pc_range=[-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]),
                 loss_cls=dict(
                     type='FocalLoss',
                     use_sigmoid=True,
                     gamma=2.0,
                     alpha=0.25,
                     loss_weight=2.0),
                 loss_bbox=dict(type='L1Loss', loss_weight=0.25),
                 loss_asso=dict(type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
                 loss_depth=None,
                 loss_iou=None,
                 loss_me=None,
                 ):
        """ Create the criterion.
        Parameters:
            num_classes: number of object categories, omitting the special no-object category
            weight_dict: dict containing as key the names of the losses and as values their relative weight.
            eos_coef: relative classification weight applied to the no-object category
        """
        super().__init__()
        self.num_classes = num_classes
        self.class_dict = class_dict
        self.assigner = build_assigner(assigner)
        self.loss_cls = build_loss(loss_cls)
        self.loss_bbox = build_loss(loss_bbox)
        self.loss_asso = build_loss(loss_asso)
        sampler_cfg = dict(type='PseudoSampler')
        self.sampler = build_sampler(sampler_cfg, context=self)
        self.bg_cls_weight = bg_cls_weight
        self.sync_cls_avg_factor = sync_cls_avg_factor
        self.register_buffer('code_weights', torch.tensor(code_weights,
                                                          requires_grad=False))

        if loss_iou is not None:
            self.loss_iou = build_loss(loss_iou)
        else:
            self.loss_iou = None
        
        if loss_me is not None:
            self.loss_me = build_loss(loss_me)
        else:
            self.loss_me = None

        if loss_depth is not None:
            if 'BCELoss' in loss_depth.type:
                self.loss_trans = loss_depth
            else:
                self.loss_trans = build_loss(loss_depth)
        else:
            self.loss_trans = None
            
        if self.loss_cls.use_sigmoid:
            self.cls_out_channels = num_classes
        else:
            self.cls_out_channels = num_classes + 1

    def clip_init(self, gt_instances: List[Instances]):
        self.gt_instances = gt_instances

    def _get_target_single(self,
                           cls_score,
                           bbox_pred,
                           gt_labels,
                           gt_bboxes,
                           gt_bboxes_ignore=None):
        """"Compute regression and classification targets for one image.
        Outputs from a single decoder layer of a single feature level are used.
        """
        num_bboxes = bbox_pred.size(0)
        # assigner and sampler
        try:
            assign_result = self.assigner.assign(bbox_pred, cls_score, gt_bboxes,
                                                gt_labels, gt_bboxes_ignore)
        except:
            print("bbox_pred:{}, cls_score:{}, gt_bboxes:{}, gt_labels:{}, gt_bboxes_ignore:{}".format(
                (bbox_pred.max(),bbox_pred.min()), (cls_score.max(),cls_score.min()),
                (gt_bboxes.max(), gt_bboxes.min()), gt_labels, gt_bboxes_ignore))
        sampling_result = self.sampler.sample(assign_result, bbox_pred,
                                              gt_bboxes)
        pos_inds = sampling_result.pos_inds
        neg_inds = sampling_result.neg_inds
        pos_gt_inds = sampling_result.pos_assigned_gt_inds

        # label targets
        labels = gt_bboxes.new_full((num_bboxes, ),
                                    self.num_classes,
                                    dtype=torch.long)
        labels[pos_inds] = gt_labels[pos_gt_inds]
        label_weights = gt_bboxes.new_ones(num_bboxes)

        # bbox targets
        bbox_targets = torch.zeros_like(bbox_pred)[..., :9]
        bbox_weights = torch.zeros_like(bbox_pred)
        bbox_weights[pos_inds] = 1.0

        # DETR
        bbox_targets[pos_inds] = sampling_result.pos_gt_bboxes
        return (labels, label_weights, bbox_targets, bbox_weights, 
                pos_inds, neg_inds, pos_gt_inds)

    def get_targets(self,
                    cls_scores_list,
                    bbox_preds_list,
                    gt_bboxes_list,
                    gt_labels_list,
                    gt_bboxes_ignore_list=None):
        """"Compute regression and classification targets for a batch image.
        Outputs from a single decoder layer of a single feature level are used.
        """
        assert gt_bboxes_ignore_list is None, \
            'Only supports for gt_bboxes_ignore setting to None.'
        num_imgs = len(cls_scores_list)
        gt_bboxes_ignore_list = [gt_bboxes_ignore_list for _ in range(num_imgs)]

        (labels_list, label_weights_list, bbox_targets_list,
         bbox_weights_list, pos_inds_list, neg_inds_list, gt_inds_list) = \
            multi_apply(self._get_target_single, cls_scores_list, bbox_preds_list,
             gt_labels_list, gt_bboxes_list, gt_bboxes_ignore_list)
        num_total_pos = sum((inds.numel() for inds in pos_inds_list))
        num_total_neg = sum((inds.numel() for inds in neg_inds_list))
        return (labels_list, label_weights_list, bbox_targets_list,
                bbox_weights_list, num_total_pos, num_total_neg, 
                pos_inds_list, gt_inds_list)

    def loss_det_single(self,
                        cls_scores,
                        bbox_preds,
                        gt_bboxes_list,
                        gt_labels_list,
                        gt_bboxes_ignore_list=None):
        """"Loss function for outputs from a single decoder layer of a single
        feature level.
        """
        num_imgs = cls_scores.size(0)
        cls_scores_list = [cls_scores[i] for i in range(num_imgs)]
        bbox_preds_list = [bbox_preds[i] for i in range(num_imgs)]
        cls_reg_targets = self.get_targets(cls_scores_list, bbox_preds_list,
                                           gt_bboxes_list, gt_labels_list, 
                                           gt_bboxes_ignore_list)
        (labels_list, label_weights_list, bbox_targets_list, bbox_weights_list,
         num_total_pos, num_total_neg, pos_inds_list, gt_inds_list) = cls_reg_targets
        labels = torch.cat(labels_list, 0)
        label_weights = torch.cat(label_weights_list, 0)
        bbox_targets = torch.cat(bbox_targets_list, 0)
        bbox_weights = torch.cat(bbox_weights_list, 0)
        pos_inds = torch.cat(pos_inds_list, 0)
        gt_inds = torch.cat(gt_inds_list, 0)

        # classification loss
        cls_scores = cls_scores.reshape(-1, self.cls_out_channels)
        # construct weighted avg_factor to match with the official DETR repo
        cls_avg_factor = num_total_pos * 1.0 + \
            num_total_neg * self.bg_cls_weight
        if self.sync_cls_avg_factor:
            cls_avg_factor = reduce_mean(
                cls_scores.new_tensor([cls_avg_factor]))

        cls_avg_factor = max(cls_avg_factor, 1)
        loss_cls = self.loss_cls(
            cls_scores, labels, label_weights, avg_factor=cls_avg_factor)

        # Compute the average number of gt boxes accross all gpus, for
        # normalization purposes
        num_total_pos = loss_cls.new_tensor([num_total_pos])
        num_total_pos = torch.clamp(reduce_mean(num_total_pos), min=1).item()

        # regression L1 loss
        bbox_preds = bbox_preds.reshape(-1, bbox_preds.size(-1))
        # normalize bboxs
        normalized_bbox_targets = normalize_bbox(bbox_targets)
        isnotnan = torch.isfinite(normalized_bbox_targets).all(dim=-1)
        bbox_weights = bbox_weights * self.code_weights

        loss_bbox = self.loss_bbox(
                bbox_preds[isnotnan, :10], normalized_bbox_targets[isnotnan, :10], 
                bbox_weights[isnotnan, :10], avg_factor=num_total_pos)

        loss_cls = torch.nan_to_num(loss_cls)
        loss_bbox = torch.nan_to_num(loss_bbox)
        return loss_cls, loss_bbox, pos_inds, gt_inds

    @force_fp32()
    def loss_det(self,
                 det_dict,
                 frame_idx):
        """"Loss function.
        """
        all_cls_scores = det_dict['pred_logits']
        all_bbox_preds = det_dict['pred_boxes']

        gt_tracklets = self.gt_instances[frame_idx]
        gt_labels_list = gt_tracklets.labels
        gt_bboxes_list = gt_tracklets.boxes
        gt_inds_list = gt_tracklets.obj_ids
        
        num_dec_layers = len(all_cls_scores)
        gt_labels_list = [gt_labels_list]
        gt_bboxes_list = [torch.cat(
            (gt_bboxes_list.gravity_center, gt_bboxes_list.tensor[:, 3:]),
            dim=1).to(gt_labels_list[0].device)]

        all_gt_bboxes_list = [gt_bboxes_list for _ in range(num_dec_layers)]
        all_gt_labels_list = [gt_labels_list for _ in range(num_dec_layers)]

        # calculate class and box loss
        losses_cls, losses_bbox, pos_inds, gt_inds = multi_apply(
            self.loss_det_single, all_cls_scores, all_bbox_preds,
            all_gt_bboxes_list, all_gt_labels_list)

        loss_dict = dict()
        # loss from the last decoder layer
        loss_dict['frame.{}.loss_cls'.format(frame_idx)] = losses_cls[-1]
        loss_dict['frame.{}.loss_bbox'.format(frame_idx)] = losses_bbox[-1]
        # loss from other decoder layers
        for layer_idx, (loss_cls_i, loss_bbox_i) in \
                    enumerate(zip(losses_cls[:-1], losses_bbox[:-1])):
            loss_dict['frame.{}.{}.loss_cls'.format(frame_idx,layer_idx)] = loss_cls_i
            loss_dict['frame.{}.{}.loss_bbox'.format(frame_idx,layer_idx)] = loss_bbox_i

        match_info = {'pos_inds': pos_inds[-1], 
                      'gt_inds': gt_inds[-1],
                      'pred_box': all_bbox_preds[-1,0],
                      'obj_ids': gt_inds_list[gt_inds[-1]]}

        track_mask = match_info['obj_ids'] >= 0        
        # select class for tracking if needed
        if self.class_dict is not None:
            all_classes = self.class_dict['all_classes']
            track_classes = self.class_dict['track_classes']            
            # arrange according to gt inds
            gt_labels = gt_labels_list[0][match_info['gt_inds']]
            cat_mask = [all_classes[_idx] in track_classes 
                          for _idx in gt_labels]
            cat_mask = torch.Tensor(cat_mask).to(dtype=bool, 
                                                 device=gt_inds_list.device)
            track_mask = track_mask & cat_mask
        match_info['pos_inds'] = match_info['pos_inds'][track_mask]
        match_info['gt_inds'] = match_info['gt_inds'][track_mask]
        match_info['obj_ids'] = match_info['obj_ids'][track_mask]
            
        return loss_dict, match_info
    
    @force_fp32()
    def loss_depth(self,
                   pred_depth,
                   gt_depth,
                   frame_idx,
                   dbound):
        """"Loss function.
        """
        # generate depth gt
        N, H, W = gt_depth.shape
        downsample_factor = H // pred_depth.shape[-2]
        depth_channel = pred_depth.shape[1]
        gt_depth = gt_depth.view(N, H // downsample_factor,
                                 downsample_factor,
                                 W // downsample_factor,
                                 downsample_factor, 1)
        gt_depth = gt_depth.permute(0, 1, 3, 5, 2, 4).contiguous()
        gt_depth = gt_depth.view(-1, downsample_factor * downsample_factor)
        gt_depth_tmp = torch.where(gt_depth == 0.0,
                                   1e5 * torch.ones_like(gt_depth),
                                   gt_depth)
        gt_depth = torch.min(gt_depth_tmp, dim=-1).values
        gt_depth = gt_depth.view(N, H // downsample_factor,
                                    W // downsample_factor)
        gt_depth = (gt_depth - (dbound[0] - dbound[2])) / dbound[2]
        gt_depth = torch.where(
            (gt_depth < depth_channel + 1) & (gt_depth >= 0.0),
             gt_depth, torch.zeros_like(gt_depth))
        # only calculate meaningful value
        gt_depth = gt_depth.reshape(-1)        
        gt_depth = F.one_hot(gt_depth.long(),
                             num_classes=depth_channel + 1).view(
                             -1, depth_channel + 1)[:, 1:]
        pred_depth = pred_depth.permute(0,2,3,1).reshape(-1, depth_channel)
        fg_mask = torch.max(gt_depth, dim=1).values > 0.0
        
        if 'BCELoss' in self.loss_trans.type:
            loss_depth = self.loss_trans.loss_weight * \
                            F.binary_cross_entropy(pred_depth[fg_mask], 
                                                gt_depth[fg_mask].float(),
                                                reduction=self.loss_trans.reduction)
        
        if self.loss_trans.reduction == 'none':
            loss_depth = loss_depth.sum() / max(1.0, fg_mask.sum())
            
        loss_dict = {}
        loss_dict['frame.{}.loss_depth'.format(frame_idx)] = loss_depth

        return loss_dict
        
    
    @force_fp32()
    def loss_track(self,
                   tracker,
                   det2track_mat,
                   match_info,
                   frame_idx,
                   zero_val=None,):
        """"Loss function for outputs from a single decoder layer of a single
        feature level.
        """
        loss_dict = {}
        # calculate per-frame tracking loss
        gt_inds = match_info['gt_inds']        
        gt_obj_ids = self.gt_instances[frame_idx].obj_ids[gt_inds]
        # not consider fp or fn tracks
        valid_track = tracker.valid_track
        track_inds = tracker.obj_idxes[valid_track]
        
        if zero_val is None:
            zero_val = torch.zeros(1, device=valid_track.device)[0]
        
        if det2track_mat is None:
            # zero_val = torch.zeros(1, device=valid_track.device)[0]
            loss_dict['frame.{}.loss_asso'.format(frame_idx)] = zero_val
            # use max entroy regulartion
            if self.loss_me is not None:
                loss_dict['frame.{}.loss_me'.format(frame_idx)] = zero_val
            return loss_dict
        
        # construct GT for each frame
        gt_per_frame = torch.full((len(det2track_mat),), -1).to(det2track_mat.device)
        for _idx, obj_id in enumerate(gt_obj_ids):
            if obj_id not in track_inds:
                continue
            gt_per_frame[_idx] = (track_inds==obj_id).nonzero(as_tuple=True)[0][0]

        # filter out new-born object
        gt_mask = (gt_per_frame >= 0)
        det2track_mat = det2track_mat[gt_mask]
        gt_per_frame = gt_per_frame[gt_mask]
        if len(det2track_mat) > 0:
            loss_single = self.loss_asso(det2track_mat, gt_per_frame.long())
        else:
            loss_single = zero_val
        
        loss_dict['frame.{}.loss_asso'.format(frame_idx)] = loss_single
        
        # debug use
        if loss_single.isnan():
            print("Asso Mat:{}, GT:{}....".format(det2track_mat, gt_per_frame))

        # use max entroy regulartion
        if self.loss_me is not None:
            track_num = det2track_mat.shape[1]
            if (track_num < 3) or (gt_mask.sum() == 0):
                loss_me = zero_val
            else:
                mask_me = F.one_hot(gt_per_frame.long(), num_classes=track_num + 1)
                mask_me = (1 - mask_me[:, :track_num]).bool()
                gt_me = torch.ones(det2track_mat.shape[0], 
                                   det2track_mat.shape[1]-1).to(det2track_mat.device)
                gt_me = gt_me / gt_me.shape[1]
                det2track_me = det2track_mat[mask_me].reshape(len(mask_me),-1)
                loss_me = self.loss_me(det2track_me, gt_me)
            loss_dict['frame.{}.loss_me'.format(frame_idx)] = loss_me

        return loss_dict