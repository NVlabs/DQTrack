# Copyright (c) 2023, NVIDIA Corporation & Affiliates. All rights reserved. 
# 
# This work is made available under the Nvidia Source Code License-NC. 
# To view a copy of this license, visit 
# https://github.com/NVlabs/DQTrack/blob/main/LICENSE
# Modified from UVTR (https://github.com/dvlab-research/UVTR)
import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import Linear, bias_init_with_prob
from mmcv.runner import force_fp32, auto_fp16
                        
from mmdet.core import (multi_apply, multi_apply, reduce_mean)
from mmdet.models.utils.transformer import inverse_sigmoid
from mmdet.models import HEADS
from mmdet.models.utils import build_transformer
from mmcv.cnn.bricks.transformer import build_positional_encoding
from ..utils import Uni3DViewTrans


@HEADS.register_module()
class UVTRTrackHead(nn.Module):
    """Head of UVTRTrack. 
    Args:
        with_box_refine (bool): Whether to refine the reference points
            in the decoder. Defaults to False.
        as_two_stage (bool) : Whether to generate the proposal from
            the outputs of encoder.
        transformer (obj:`ConfigDict`): ConfigDict is used for building
            the Encoder and Decoder.
    """
    def __init__(self,
                 num_classes,
                 in_channels,
                 unified_conv=None,
                 view_cfg=None,
                 with_box_refine=False,
                 with_size_refine=False,
                 transformer=None,
                 num_cls_fcs=2,
                 num_reg_fcs=2,
                 code_weights=None,
                 positional_encoding=None,
                 **kwargs):
        super(UVTRTrackHead, self).__init__()

        self.with_box_refine = with_box_refine
        self.with_size_refine = with_size_refine
        if 'code_size' in kwargs:
            self.code_size = kwargs['code_size']
        else:
            self.code_size = 10
        if code_weights is not None:
            self.code_weights = code_weights
        else:
            self.code_weights = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.2, 0.2]

        self.pc_range = view_cfg.pc_range
        self.in_channels = in_channels
        self.num_cls_fcs = num_cls_fcs - 1
        self.num_reg_fcs = num_reg_fcs
        self.code_weights = nn.Parameter(torch.tensor(
            self.code_weights, requires_grad=False), requires_grad=False)
        self.unified_conv = unified_conv
        # transformer config
        self.transformer = build_transformer(transformer)
        self.positional_encoding = build_positional_encoding(positional_encoding)
        self.embed_dims = self.transformer.embed_dims
        self.cls_out_channels = num_classes

        if view_cfg is not None:
            self.view_trans = Uni3DViewTrans(**view_cfg)
        
        self.fp16_enabled = False      
        if self.unified_conv is not None:
            self.conv_layer = []
            for k in range(self.unified_conv['num_conv']):
                conv = nn.Sequential(
                            nn.Conv3d(view_cfg['embed_dims'],
                                    view_cfg['embed_dims'],
                                    kernel_size=3,
                                    stride=1,
                                    padding=1,
                                    bias=True),
                            nn.BatchNorm3d(view_cfg['embed_dims']),
                            nn.ReLU(inplace=True))
                self.add_module("{}_head_{}".format('conv_trans', k + 1), conv)
                self.conv_layer.append(conv)

        self._init_layers()

    def _init_layers(self):
        """Initialize classification branch and regression branch of head."""
        cls_branch = []
        for _ in range(self.num_reg_fcs):
            cls_branch.append(Linear(self.embed_dims, self.embed_dims))
            cls_branch.append(nn.LayerNorm(self.embed_dims))
            cls_branch.append(nn.ReLU(inplace=True))
        cls_branch.append(Linear(self.embed_dims, self.cls_out_channels))
        fc_cls = nn.Sequential(*cls_branch)

        reg_branch = []
        for _ in range(self.num_reg_fcs):
            reg_branch.append(Linear(self.embed_dims, self.embed_dims))
            reg_branch.append(nn.ReLU())
        reg_branch.append(Linear(self.embed_dims, self.code_size))
        reg_branch = nn.Sequential(*reg_branch)

        def _get_clones(module, N):
            return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

        # last reg_branch is used to generate proposal from
        num_pred = self.transformer.decoder.num_layers

        if self.with_box_refine:
            self.cls_branches = _get_clones(fc_cls, num_pred)
            self.reg_branches = _get_clones(reg_branch, num_pred)
        else:
            self.cls_branches = nn.ModuleList(
                [fc_cls for _ in range(num_pred)])
            self.reg_branches = nn.ModuleList(
                [reg_branch for _ in range(num_pred)])

    def init_weights(self):
        """Initialize weights of the DeformDETR head."""
        self.transformer.init_weights()
        bias_init = bias_init_with_prob(0.01)
        for m in self.cls_branches:
            nn.init.constant_(m[-1].bias, bias_init)

    @auto_fp16(apply_to=("pts_feats", "img_feats", "img_depth"))
    def forward(self, query_embeds, ref_points, ref_size, 
                pts_feats, img_feats, img_metas, img_depth):
        """Forward function.
        Args:
            mlvl_feats (tuple[Tensor]): Features from the upstream
                network, each is a 5D-tensor with shape
                (B, N, C, H, W).
        Returns:
            all_cls_scores (Tensor): Outputs from the classification head, \
                shape [nb_dec, bs, num_query, cls_out_channels]. Note \
                cls_out_channels should includes background.
            all_bbox_preds (Tensor): Sigmoid outputs from the regression \
                head with normalized coordinate format (cx, cy, w, l, cz, h, theta, vx, vy). \
                Shape [nb_dec, bs, num_query, 9].
        """
        with_image, with_point = True, True
        if img_feats is None:
            with_image = False
        elif isinstance(img_feats, dict) and img_feats['key'] is None:
            with_image = False

        if pts_feats is None:
            with_point = False
        elif isinstance(pts_feats, dict) and pts_feats['key'] is None:
            with_point = False
            pts_feats = None

        # transfer to voxel level
        if with_image:
            img_feats = self.view_trans(img_feats, img_metas=img_metas, img_depth=img_depth)
        # shape: (N, L, C, D, H, W)
        if with_point:
            if len(pts_feats.shape) == 5:
                pts_feats = pts_feats.unsqueeze(1)

        if self.unified_conv is not None:
            raw_shape = pts_feats.shape
            unified_feats = pts_feats.flatten(1,2) + img_feats.flatten(1,2)
            for layer in self.conv_layer:
                unified_feats = layer(unified_feats)
            img_feats = unified_feats.reshape(*raw_shape)
            pts_feats = None

        hs, init_reference, inter_references, inter_box_sizes = self.transformer(
            pts_feats,
            img_feats,
            query_embeds,
            ref_points,
            ref_size,
            reg_branches=self.reg_branches if self.with_box_refine else None,  # noqa:E501
            img_metas=img_metas,
        )
        hs = hs.permute(0, 2, 1, 3)
        outputs_classes = []
        outputs_coords = []

        for lvl in range(hs.shape[0]):
            if lvl == 0:
                reference = init_reference
                ref_size_base = ref_size
            else:
                reference = inter_references[lvl - 1]
                if self.with_size_refine and ref_size is not None:
                    ref_size_base = inter_box_sizes[lvl - 1]
            reference = inverse_sigmoid(reference)
            outputs_class = self.cls_branches[lvl](hs[lvl])
            xywlzh = self.reg_branches[lvl](hs[lvl])

            # TODO: check the shape of reference
            assert reference.shape[-1] == 3
            xywlzh[..., 0:2] += reference[..., 0:2]
            xywlzh[..., 0:2] = xywlzh[..., 0:2].sigmoid()
            xywlzh[..., 4:5] += reference[..., 2:3]
            xywlzh[..., 4:5] = xywlzh[..., 4:5].sigmoid()

            ref_points = torch.cat([xywlzh[...,0:2],xywlzh[...,4:5]], dim=-1)

            # transfer to lidar system
            xywlzh[..., 0:1] = (xywlzh[..., 0:1] * (self.pc_range[3] - self.pc_range[0]) + self.pc_range[0])
            xywlzh[..., 1:2] = (xywlzh[..., 1:2] * (self.pc_range[4] - self.pc_range[1]) + self.pc_range[1])
            xywlzh[..., 4:5] = (xywlzh[..., 4:5] * (self.pc_range[5] - self.pc_range[2]) + self.pc_range[2])

            if self.with_size_refine and ref_size_base is not None:
                xywlzh[..., 2:4] = xywlzh[..., 2:4] + ref_size_base[..., 0:2]
                xywlzh[..., 5:6] = xywlzh[..., 5:6] + ref_size_base[..., 2:3]

            # TODO: check if using sigmoid
            outputs_coord = xywlzh
            outputs_classes.append(outputs_class)
            outputs_coords.append(outputs_coord)

        outputs_classes = torch.stack(outputs_classes)
        outputs_coords = torch.stack(outputs_coords)
        last_query_feat = hs[-1]
        last_ref_points = inverse_sigmoid(ref_points)

        outs = {
            'all_cls_scores': outputs_classes,
            'all_bbox_preds': outputs_coords,
            'last_query_feat': last_query_feat,
            'last_ref_points': last_ref_points,
        }

        return outs
