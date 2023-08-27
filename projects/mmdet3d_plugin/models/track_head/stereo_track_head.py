# Copyright (c) 2023, NVIDIA Corporation & Affiliates. All rights reserved. 
# 
# This work is made available under the Nvidia Source Code License-NC. 
# To view a copy of this license, visit 
# https://github.com/NVlabs/DQTrack/blob/main/LICENSE
# Modified from BEVStereo (https://github.com/Megvii-BaseDetection/BEVStereo)
import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from mmcv.cnn import Linear, bias_init_with_prob
from mmcv.runner import force_fp32, auto_fp16
                        
from mmdet.core import (multi_apply, multi_apply, reduce_mean)
from mmdet.models.utils.transformer import inverse_sigmoid
from mmdet.models import HEADS
from mmdet.models.utils import build_transformer
from mmcv.cnn.bricks.transformer import build_positional_encoding
from mmdet3d.models import builder
from ..utils import Uni3DViewTrans, StereoViewTrans


@HEADS.register_module()
class StereoTrackHead(nn.Module):
    """Head of StereoTrack. 
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
                 trans_type=None,
                 view_cfg=None,
                 with_box_refine=False,
                 with_size_refine=False,
                 bev_backbone=None,
                 bev_neck=None,
                 transformer=None,
                 num_cls_fcs=2,
                 num_reg_fcs=2,
                 code_weights=None,
                 positional_encoding=None,
                 multi_frame=dict(),
                 **kwargs):
        super(StereoTrackHead, self).__init__()

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

        
        self.in_channels = in_channels
        self.num_cls_fcs = num_cls_fcs - 1
        self.num_reg_fcs = num_reg_fcs
        self.code_weights = nn.Parameter(torch.tensor(
            self.code_weights, requires_grad=False), requires_grad=False)
        self.unified_conv = unified_conv
        
        
        self.multi_frame = multi_frame
        self.frame_feat = None
        self.pre_trans_mat = None
        # transformer config
        self.transformer = build_transformer(transformer)
        self.positional_encoding = build_positional_encoding(positional_encoding)
        self.embed_dims = self.transformer.embed_dims
        self.cls_out_channels = num_classes
        self.pc_range = view_cfg.pc_range

        if view_cfg.get('use_image', True):
            self.use_temporal = view_cfg.get('use_temporal', False)
            self.detach_grad = view_cfg.get('detach_grad', True)

            if 'Stereo' in trans_type:
                self.view_trans = StereoViewTrans(**view_cfg)
            elif 'Uni3D' in trans_type:
                self.view_trans = Uni3DViewTrans(**view_cfg)
        
            if self.use_temporal:
                self.trans_conv = nn.Conv2d(2*in_channels, self.embed_dims, 1)
            else:
                self.trans_conv = None
        
        if view_cfg.get('use_pts', False):
            self.trans_conv_pts = nn.Conv2d(view_cfg.pts_channel, self.embed_dims, 1)
        
        if bev_backbone is not None:
            self.bev_backbone = builder.build_backbone(bev_backbone)
            del self.bev_backbone.maxpool
        else:
            self.bev_backbone = None
        
        if bev_neck is not None:
            self.bev_neck = builder.build_neck(bev_neck)
            in_channels = sum(self.bev_neck.out_channels)
            out_channels = in_channels // len(self.bev_neck.out_channels)
            self.neck_trans = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        else:
            self.bev_neck = None

        self.fp16_enabled = False      
        if self.unified_conv is not None:
            self.conv_layer = []
            in_channels = self.unified_conv.get('in_channels', self.embed_dims)
            kernel_size = self.unified_conv.get('kernel_size', 3)
            for k in range(self.unified_conv['num_conv']):
                conv = nn.Sequential(
                            nn.Conv2d(in_channels,
                                      self.embed_dims,
                                      kernel_size=kernel_size,
                                      stride=1,
                                      padding=(kernel_size-1)//2,
                                      bias=False),
                            nn.BatchNorm2d(self.embed_dims),
                            nn.ReLU(inplace=True))
                in_channels = self.embed_dims
                self.add_module("{}_head_{}".format('conv_trans', k + 1), conv)
                self.conv_layer.append(conv)

        bev_grid_x = torch.linspace(self.pc_range[0], 
                                    self.pc_range[3], 
                                    view_cfg.voxel_shape[0]).view(1,-1).expand(*view_cfg.voxel_shape[:2])
        bev_grid_y = torch.linspace(self.pc_range[1], 
                                    self.pc_range[4], 
                                    view_cfg.voxel_shape[1]).view(-1,1).expand(*view_cfg.voxel_shape[:2])
        bev_grid_z = torch.ones_like(bev_grid_y)
        paddings = torch.ones_like(bev_grid_y)
        self.bev_grid = torch.stack([bev_grid_x, bev_grid_y, bev_grid_z, paddings], dim=-1)
        
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

        if self.bev_backbone is not None:
            self.bev_backbone.init_weights()
        
        if self.bev_neck is not None:
            self.bev_neck.init_weights()

        # combine multi-frame with conv
        if 'conv' in self.multi_frame.get('type', ''):
            self.frame_trans = nn.Conv2d(self.multi_frame.frame_num * self.embed_dims, 
                                         self.embed_dims,
                                         kernel_size=1)


    def init_weights(self):
        """Initialize weights of the DeformDETR head."""
        self.transformer.init_weights()
        bias_init = bias_init_with_prob(0.01)
        for m in self.cls_branches:
            nn.init.constant_(m[-1].bias, bias_init)

    def stereo_trans(self, img_feats, img_metas, img_depth):
        lidar2img, lidar2ego = [], []
        img_aug_mat, bev_aug_mat, stereo_feat = [], [], []
        for img_meta in img_metas:
            lidar2img.append(img_meta['lidar2img'])
            lidar2ego.append(img_meta['lidar2ego'])
            if 'img_aug_mat' in img_meta:
                img_aug_mat.append(torch.stack(img_meta['img_aug_mat'], dim=0))
            if 'uni_rot_aug' in img_meta:
                bev_aug_mat.append(torch.stack(img_meta['uni_rot_aug'], dim=0))
        # calculate batch-wise key2sweep image matrix
        lidar2img = torch.tensor(np.array(lidar2img))
        lidar2ego = torch.tensor(np.array(lidar2ego))
        lidar2img = lidar2img.reshape(-1, *lidar2img.shape[-4:])
        lidar2ego = lidar2ego.reshape(-1, *lidar2ego.shape[-2:])
        if len(img_aug_mat) > 0:
            img_aug_mat = torch.stack(img_aug_mat, dim=0).to(lidar2img)
            img_aug_mat = img_aug_mat.reshape(-1, *lidar2img.shape[1:])
        if len(bev_aug_mat) > 0:
            bev_aug_mat = torch.stack(bev_aug_mat, dim=0)
        if self.use_temporal:
            temp_dict = ['key', 'sweep']
        else:
            temp_dict = ['key']

        # temporal stereo use 2 images
        for ref_key in temp_dict:
            if ref_key == 'key':
                ref_dict = img_depth['ref_depth']
                src_dict = img_depth['src_depth']
                ref_dict['ref_stereo'] = img_feats['ref_stereo']
                src_dict['src_stereo'] = img_feats['src_stereo']
                lidar2img_key = lidar2img[:,:,0]
                lidar2img_sweep = lidar2img[:,:,1]
                if len(img_aug_mat) > 0:
                    img_aug_mat_key = img_aug_mat[:,:,0]
                    img_aug_mat_sweep = img_aug_mat[:,:,1]
            else:
                ref_dict = img_depth['src_depth']
                src_dict = img_depth['ref_depth']
                ref_dict['ref_stereo'] = img_feats['src_stereo']
                src_dict['src_stereo'] = img_feats['ref_stereo']
                lidar2img_key = lidar2img[:,:,1]
                lidar2img_sweep = lidar2img[:,:,0]
                if len(img_aug_mat) > 0:
                    img_aug_mat_key = img_aug_mat[:,:,1]
                    img_aug_mat_sweep = img_aug_mat[:,:,0]
            
            # both key and sweep images are aligned to key lidar
            key2sweep_img = lidar2img_sweep @ lidar2img_key.inverse()
            key2sweep_img = key2sweep_img.reshape(-1, *key2sweep_img.shape[-2:])
            trans_dict = dict(
                key2sweep_img = key2sweep_img.to(ref_dict['mu'])
            )

            if len(img_aug_mat) > 0:
                img_aug_mat_key = img_aug_mat_key.reshape(-1, *img_aug_mat_key.shape[-2:])
                img_aug_mat_sweep = img_aug_mat_sweep.reshape(-1, *img_aug_mat_sweep.shape[-2:])
                trans_dict['img_aug_mat_key'] = img_aug_mat_key.to(ref_dict['mu'])
                trans_dict['img_aug_mat_sweep'] = img_aug_mat_sweep.to(ref_dict['mu'])
            if len(bev_aug_mat) > 0:
                trans_dict['bev_aug_mat'] = bev_aug_mat.to(ref_dict['mu'])
            
            # trans img coord to lidar system
            trans_dict['sensor2img_key'] = lidar2img_key

            if ref_key == 'key':
                trans_feat, pred_depth = self.view_trans(ref_dict, src_dict, trans_dict, img_metas, return_depth=True)
            else:
                if self.detach_grad:
                    with torch.no_grad():
                        trans_feat = self.view_trans(ref_dict, src_dict, trans_dict, img_metas)
                else:
                    trans_feat = self.view_trans(ref_dict, src_dict, trans_dict, img_metas)

            stereo_feat.append(trans_feat)

        stereo_feat = torch.cat(stereo_feat, dim=1)
        del ref_dict, src_dict

        return stereo_feat, pred_depth

    def bev_encoder(self, img_feats):
        bev_outs = [img_feats]

        # use backbone for bev features
        if self.bev_backbone.deep_stem:
            img_feats = self.bev_backbone.stem(img_feats)
        else:
            img_feats = self.bev_backbone.conv1(img_feats)
            img_feats = self.bev_backbone.norm1(img_feats)
            img_feats = self.bev_backbone.relu(img_feats)
        for i, layer_name in enumerate(self.bev_backbone.res_layers):
            res_layer = getattr(self.bev_backbone, layer_name)
            img_feats = res_layer(img_feats)
            if i in self.bev_backbone.out_indices:
                bev_outs.append(img_feats)

        # use neck for bev features
        if self.bev_neck is not None:
            img_feats = self.bev_neck(bev_outs)[0]
            img_feats = self.neck_trans(img_feats)

        return img_feats

    def frame_update(self, img_feats, img_metas):
        """Frame update
        """
        if len(self.multi_frame.get('type', '')) == 0:
            return img_feats

        if self.training:
            img_feats_mf = []
            # update frame-by-frame
            for _idx in range(len(img_feats)):
                cur_trans_mat = torch.Tensor(img_metas[0]['lidar2global'][_idx]
                                             ).to(img_feats.device)
                if _idx == 0:
                    new_feat = img_feats[_idx]
                else:
                    # calculate current to previous matrix
                    cur2pre_mat = self.pre_trans_mat.inverse() @ cur_trans_mat
                    pre_bev_grid = self.bev_grid.to(img_feats.device) @ cur2pre_mat.T
                    pre_bev_grid[...,0:1] = (pre_bev_grid[...,0:1] - self.pc_range[0]
                                             ) / (self.pc_range[3] - self.pc_range[0])
                    pre_bev_grid[...,1:2] = (pre_bev_grid[...,1:2] - self.pc_range[1]
                                             ) / (self.pc_range[4] - self.pc_range[1])
                    pre_bev_grid = (pre_bev_grid[...,:2] - 0.5) * 2
                    
                    if 'ema' in self.multi_frame.type:
                        pre_feat = img_feats_mf[_idx-1]
                        if self.multi_frame.get('detach', False):
                            pre_feat = pre_feat.detach()
                        
                        pre_feat = F.grid_sample(pre_feat[None], 
                                                 pre_bev_grid[None], 
                                                 mode='bilinear', 
                                                 padding_mode='zeros')[0]
                        
                        new_feat = self.multi_frame.ema_decay * img_feats[_idx] + \
                                (1-self.multi_frame.ema_decay) * pre_feat
                    elif 'conv' in self.multi_frame.type:
                        if 'update' in self.multi_frame.type:
                            pre_feat = img_feats_mf[_idx-1]
                        else:
                            pre_feat = img_feats[_idx-1]

                        if self.multi_frame.get('detach', False):
                            pre_feat = pre_feat.detach()
                        
                        pre_feat = F.grid_sample(pre_feat[None], 
                                                 pre_bev_grid[None], 
                                                 mode='bilinear', 
                                                 padding_mode='zeros')[0]
                        
                        new_feat = torch.cat([pre_feat, img_feats[_idx]])
                        new_feat = self.frame_trans(new_feat[None])[0]
                img_feats_mf.append(new_feat)
                self.pre_trans_mat = cur_trans_mat
            img_feats = torch.stack(img_feats_mf, dim=0)
        else:            
            cur_trans_mat = torch.Tensor(img_metas[0]['lidar2global'][0]).to(img_feats.device)
            if (self.frame_feat is not None) and (not img_metas[0]['scene_change']):
                # calculate current to previous matrix
                cur2pre_mat = self.pre_trans_mat.inverse() @ cur_trans_mat
                pre_bev_grid = self.bev_grid.to(img_feats.device) @ cur2pre_mat.T
                pre_bev_grid[...,0:1] = (pre_bev_grid[...,0:1] - self.pc_range[0]
                                        ) / (self.pc_range[3] - self.pc_range[0])
                pre_bev_grid[...,1:2] = (pre_bev_grid[...,1:2] - self.pc_range[1]
                                        ) / (self.pc_range[4] - self.pc_range[1])
                pre_bev_grid = (pre_bev_grid[...,:2] - 0.5) * 2
                
                pre_feat = F.grid_sample(self.frame_feat, 
                                         pre_bev_grid[None], 
                                         mode='bilinear', 
                                         padding_mode='zeros')
                
                if 'ema' in self.multi_frame.type:
                    img_feats = self.multi_frame.ema_decay * img_feats + \
                                (1-self.multi_frame.ema_decay) * pre_feat
                elif 'conv' in self.multi_frame.type:
                    raw_feats = img_feats.clone()
                    new_feat = torch.cat([pre_feat, img_feats]).flatten(0,1)
                    img_feats = self.frame_trans(new_feat[None])
            else:
                if 'conv' in self.multi_frame.type:
                    raw_feats = img_feats.clone()
            
            if 'update' in self.multi_frame.type:
                self.frame_feat = img_feats
            elif 'conv' in self.multi_frame.type:
                self.frame_feat = raw_feats
        
            self.pre_trans_mat = cur_trans_mat
        
        return img_feats

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
        elif isinstance(img_feats, dict) and 'key' in img_feats:
            if img_feats['key'] is None:
                with_image = False

        if pts_feats is None:
            with_point = False
        elif isinstance(pts_feats, dict) and 'key' in pts_feats:
            if pts_feats['key'] is None:
                with_point = False
                pts_feats = None

        # transfer to voxel level
        if with_image:
            img_feats, pred_depth = self.stereo_trans(img_feats, img_metas=img_metas, img_depth=img_depth)
            if self.trans_conv:
                img_feats = self.trans_conv(img_feats)

        if with_point:
            batch_num = len(pts_feats)
            pts_feats = self.trans_conv_pts(torch.cat(pts_feats, dim=0))
        else:
            batch_num = len(img_metas)
        
        if self.unified_conv is not None:
            if pts_feats is not None and img_feats is not None:
                unified_feats = pts_feats + img_feats
            elif pts_feats is not None:
                unified_feats = pts_feats
            else:
                unified_feats = img_feats
            
            for layer in self.conv_layer:
                unified_feats = layer(unified_feats)
            
            img_feats = unified_feats
            pts_feats = None

        # use backbone for bev features
        if self.bev_backbone is not None:
            img_feats = self.bev_encoder(img_feats)
        
        if batch_num > 1:
            img_feats = img_feats.reshape(batch_num, -1, *img_feats.shape[1:])
            img_feats = [self.frame_update(img_feats[_idx], [img_metas[_idx]]) 
                         for _idx in range(len(img_feats))]
            img_feats = torch.cat(img_feats, dim=0)
        else:
            # use multi-frame
            img_feats = self.frame_update(img_feats, img_metas)
        
        outs = {'raw_feat':img_feats.clone()}
        
        # shape: (B, S, C, H, W)
        img_feats = img_feats[:, None]
        if with_image:
            pred_depth = pred_depth.reshape(len(img_feats), -1, *pred_depth.shape[1:])
        else:
            pred_depth = None
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

        outs.update({
            'all_cls_scores': outputs_classes,
            'all_bbox_preds': outputs_coords,
            'last_query_feat': last_query_feat,
            'last_ref_points': last_ref_points,
            'pred_depth': pred_depth,
        })

        return outs
