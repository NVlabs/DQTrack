# Copyright (c) 2023, NVIDIA Corporation & Affiliates. All rights reserved. 
# 
# This work is made available under the Nvidia Source Code License-NC. 
# To view a copy of this license, visit 
# https://github.com/NVlabs/DQTrack/blob/main/LICENSE
# Modified from BEVStereo (https://github.com/Megvii-BaseDetection/BEVStereo)
import numpy as np
import torch
import math
import torch.nn as nn
import torch.nn.functional as F
from scipy.special import erf
from scipy.stats import norm
from mmcv.runner import force_fp32, auto_fp16
from mmcv.cnn import xavier_init, build_norm_layer
from mmcv.runner.base_module import BaseModule
from mmdet.models.backbones.resnet import BasicBlock
from projects.mmdet3d_plugin.models.utils.track_dq_utils import ConvBnReLU3D
from projects.mmdet3d_plugin.models.ops.voxel_pooling import voxel_pooling

class StereoViewTrans(BaseModule):
    """Implements the view transformer.
    """
    def __init__(self,
                 num_cams=6,
                 num_sweeps=1,
                 num_samples=3,
                 num_groups=8,
                 em_iter=3,
                 use_wnet=True,
                 depth_bound=[2.0, 58.0, 0.5],
                 range_list=[[2, 8], [8, 16], [16, 28], [28, 58]],
                 sampling_range=3,
                 downsample_factor=16,
                 stereo_downsample_factor=4,
                 pc_range=[-51.2, -51.2, -5.0, 51.2, 51.2, 3.0],
                 voxel_size=[0.8, 0.8, 8],
                 voxel_shape=[128, 128, 1],
                 **kwargs):
        super(StereoViewTrans, self).__init__()
        self.conv_layer = []
        
        self.num_cams = num_cams
        self.num_sweeps = num_sweeps
        self.num_groups = num_groups
        self.em_iter = em_iter
        self.use_wnet = use_wnet
        self.depth_bound = depth_bound
        self.range_list = range_list
        self.sampling_range = sampling_range
        self.num_samples = num_samples
        self.depth_channel = int((depth_bound[1] - depth_bound[0])//depth_bound[2]),
        self.downsample_factor = downsample_factor
        self.stereo_downsample_factor = stereo_downsample_factor
        self.pc_range = torch.tensor(pc_range, dtype=torch.float)
        self.voxel_size = torch.tensor(voxel_size, dtype=torch.float)
        self.voxel_shape = torch.tensor(voxel_shape, dtype=torch.long)
        self.k_list = self.depth_sampling()
        self.frustum = None
        self.depth_frustum = None
        # self.sweep_fusion = sweep_fusion.get("type", "")
        self.similarity_net = nn.Sequential(
            ConvBnReLU3D(in_channels=num_groups,
                         out_channels=16,
                         kernel_size=1,
                         stride=1,
                         pad=0),
            ConvBnReLU3D(in_channels=16,
                         out_channels=8,
                         kernel_size=1,
                         stride=1,
                         pad=0),
            nn.Conv3d(in_channels=8,
                      out_channels=1,
                      kernel_size=1,
                      stride=1,
                      padding=0))

        self.downsample_net = nn.Sequential(
            nn.Conv2d(self.depth_channel[0], 256, 3, 2, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, 3, 2, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, self.depth_channel[0], 1, 1, 0))

        if self.use_wnet:
            self.weight_net = nn.Sequential(
                nn.Conv2d(224, 64, 3, 1, 1),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                BasicBlock(64, 64),
                BasicBlock(64, 64),
                nn.Conv2d(64, 1, 1, 1, 0),
                nn.Sigmoid())

        self.init_weights()
        if kwargs.get("fp16_enabled", False):
            self.fp16_enabled = True

    def init_weights(self):
        """Default initialization for Parameters of Module."""
        for layer in self.conv_layer:
            xavier_init(layer, distribution='uniform', bias=0.)

    def depth_sampling(self):
        """Generate sampling range of candidates.

        Returns:
            list[float]: List of all candidates.
        """
        P_total = erf(self.sampling_range / np.sqrt(2))  # Probability covered by the sampling range
        idx_list = np.arange(0, self.num_samples + 1)
        p_list = (1 - P_total) / 2 + ((idx_list / self.num_samples) * P_total)
        k_list = norm.ppf(p_list)
        k_list = (k_list[1:] + k_list[:-1]) / 2
        return torch.Tensor(list(k_list))

    def create_frustum(self, img_metas, device):
        """Generate frustum"""
        # make grid in image plane
        ogfH, ogfW = img_metas[0]['img_shape'][0][0][:2]
        fH, fW = ogfH // self.downsample_factor, ogfW // self.downsample_factor
        d_coords = torch.arange(*self.depth_bound, 
                        dtype=torch.float).view(-1, 1, 1).expand(-1, fH, fW)
        num_depth = d_coords.shape[0]
        x_coords = torch.linspace(0, ogfW - 1, fW, 
                        dtype=torch.float).view(1, 1, fW).expand(num_depth, fH, fW)
        y_coords = torch.linspace(0, ogfH - 1, fH,
                        dtype=torch.float).view(1, fH, 1).expand(num_depth, fH, fW)
        paddings = torch.ones_like(d_coords)
        # D x H x W x 4
        frustum = torch.stack((x_coords, y_coords, d_coords, paddings), -1)
        return frustum.to(device)
        

    def create_depth_sample_frustum(self, depth_sample, img_metas, downsample_factor=16):
        """Generate frustum"""
        # make grid in image plane
        ogfH, ogfW = img_metas[0]['img_shape'][0][0][:2]
        fH, fW = ogfH // downsample_factor, ogfW // downsample_factor
        batch_size, num_depth, _, _ = depth_sample.shape
        x_coords = torch.linspace(0, ogfW - 1, fW, dtype=torch.float,
                                  device=depth_sample.device)
        x_coords = x_coords.view(1, 1, 1, fW).expand(batch_size, num_depth, fH, fW)
        y_coords = torch.linspace(0, ogfH - 1, fH, dtype=torch.float,
                                  device=depth_sample.device)
        y_coords = y_coords.view(1, 1, fH, 1).expand(batch_size, num_depth, fH, fW)
        paddings = torch.ones_like(depth_sample)
        # D x H x W x 4
        frustum = torch.stack([x_coords, y_coords, depth_sample, paddings], -1)
        return frustum

    def homo_warping(self, src_stereo, trans_dict, depth_sample, frustum, eps=1e-3):
        """Used for mvs method to transfer sweep image feature to
           key image feature.
        """
        batch_size, num_channels, height, width = src_stereo.shape
        num_depth = frustum.shape[1]
        key2sweep_img = trans_dict['key2sweep_img']
        with torch.no_grad():
            # prepare for inverse image aug
            frustum[..., 2] = 1
            # undo image-level aug for key frame
            if 'img_aug_mat_key' in trans_dict:
                img_aug_mat_key = trans_dict['img_aug_mat_key'].float()
                img_aug_mat_key = img_aug_mat_key.inverse().to(frustum)
                img_aug_mat_key = img_aug_mat_key.permute(0,2,1).reshape(-1, 1, 1, 4, 4)
                frustum = frustum @ img_aug_mat_key

            frustum[..., :3] *= depth_sample[...,None]
            key2sweep_img = key2sweep_img.permute(0,2,1).reshape(-1, 1, 1, 4, 4)
            # transfer key points to sweep points
            point_sweep = frustum @ key2sweep_img
            point_sweep[..., :2] = point_sweep[..., :2] / torch.maximum(
                point_sweep[...,2:3], eps * torch.ones_like(point_sweep[...,2:3]))
            
            # do image-level aug for sweep frame
            if 'img_aug_mat_sweep' in trans_dict:
                img_aug_mat_sweep = trans_dict['img_aug_mat_sweep']
                img_aug_mat_sweep = img_aug_mat_sweep.permute(0,2,1).reshape(-1, 1, 1, 4, 4)
                point_sweep = point_sweep @ img_aug_mat_sweep

            neg_mask = (point_sweep[..., 2] < eps)
            point_sweep[..., 0][neg_mask] = width * self.stereo_downsample_factor
            point_sweep[..., 1][neg_mask] = height * self.stereo_downsample_factor
            # normalize point
            point_sweep[..., 0] /= (width * self.stereo_downsample_factor - 1)
            point_sweep[..., 1] /= (height * self.stereo_downsample_factor - 1)
            point_sweep = (point_sweep[..., :2] - 0.5) * 2
        
        src_feat = F.grid_sample(
            src_stereo, 
            point_sweep.view(batch_size, num_depth*height, width, 2),
            mode='bilinear',
            padding_mode='zeros'
        )

        del point_sweep
        
        return src_feat.reshape(batch_size, num_channels, num_depth, height, width)

    def generate_cost_volume(self, ref_stereo, src_stereo, trans_dict, depth_sample, depth_sample_frustum):
        """Generate cost volume based on depth sample.
        """
        batch_size, num_channels, height, width = ref_stereo.shape

        src_stereo_warp = self.homo_warping(src_stereo, 
                                            trans_dict, 
                                            depth_sample, 
                                            depth_sample_frustum)

        src_stereo_warp = src_stereo_warp.reshape(batch_size, self.num_groups, 
                                                  num_channels//self.num_groups,
                                                  self.num_samples, height, width)
        ref_stereo = ref_stereo.reshape(batch_size, self.num_groups, 
                                        num_channels//self.num_groups,
                                        height, width)

        cost_volume = torch.mean(ref_stereo.unsqueeze(3) * src_stereo_warp, dim=2)
        depth_score = self.similarity_net(cost_volume).squeeze(1)

        del cost_volume

        return depth_score

    def forward_stereo(self, ref_dict, src_dict, trans_dict, img_metas, min_sigma=1, eps=1e-6):
        """Calculate stereo depth.
        """
        ref_stereo = ref_dict['ref_stereo']
        src_stereo = src_dict['src_stereo']
        ref_range_score = ref_dict['range_score'].softmax(1)
        mu_all_sweeps = [ref_dict['mu'], src_dict['mu']]
        sigma_all_sweeps = [ref_dict['sigma'], src_dict['sigma']]
        B, _, H, W = ref_stereo.shape
        d_coords = torch.arange(*self.depth_bound,
                                dtype=torch.float,
                                device=ref_stereo.device).reshape(1, -1, 1, 1)
        d_coords = d_coords.repeat(B, 1, H, W)

        stereo_depth = ref_stereo.new_zeros(B, self.depth_channel[0], H, W)
        for range_idx in range(ref_range_score.shape[1]):
            # map mu to the corresponding interval.
            range_start = self.range_list[range_idx][0]
            mu_single_range = [
                mu[:, range_idx:range_idx + 1].sigmoid() *
                (self.range_list[range_idx][1] - self.range_list[range_idx][0])
                + range_start for mu in mu_all_sweeps
            ]
            sigma_single_range = [
                sigma[:, range_idx:range_idx + 1]
                for sigma in sigma_all_sweeps
            ]
            ref_mu = mu_single_range[0]
            ref_sigma = sigma_single_range[0]
            for _ in range(self.em_iter):
                depth_sample = torch.cat([ref_mu + ref_sigma * k for k in self.k_list], 1)
                if self.depth_frustum is None:
                    self.depth_frustum = self.create_depth_sample_frustum(
                            depth_sample, img_metas, self.stereo_downsample_factor)
                else:
                    self.depth_frustum[..., 2] = depth_sample
                depth_sample_frustum = self.depth_frustum.clone()

                mu_score = self.generate_cost_volume(ref_stereo, 
                                                     src_stereo, 
                                                     trans_dict,
                                                     depth_sample,
                                                     depth_sample_frustum)
                
                mu_score = mu_score.softmax(1)
                scale_factor = torch.clamp(
                    0.5 / (1e-4 + mu_score[:, self.num_samples//2:self.num_samples//2 + 1]),
                    min=0.1, max=10)
                ref_sigma = torch.clamp(ref_sigma * scale_factor, min=0.1, max=10)
                ref_mu = (depth_sample * mu_score).sum(1, keepdim=True)
                del depth_sample, depth_sample_frustum, mu_score

            ref_mu = torch.clamp(ref_mu, 
                                 max=self.range_list[range_idx][1],
                                 min=self.range_list[range_idx][0])
            range_length = int((self.range_list[range_idx][1] - self.range_list[range_idx][0])
                                // self.depth_bound[2])
        
            if self.use_wnet:
                raise NotImplementedError("Function not implemented!")
            
            ref_sigma = torch.clamp(ref_sigma, min_sigma)
            mu_repeat = ref_mu.repeat(1, range_length, 1, 1)
            depth_score_single = (-1 / 2 * ((d_coords[:, int((range_start - self.depth_bound[0]) //
                              self.depth_bound[2]):range_length + int((range_start - self.depth_bound[0]) //
                              self.depth_bound[2]), ..., ] - mu_repeat) / torch.sqrt(ref_sigma))**2)
            depth_score_single = depth_score_single.exp() / (ref_sigma * math.sqrt(2 * math.pi) + eps)
            stereo_depth[:, int((range_start - self.depth_bound[0]) // self.depth_bound[2])
                         : range_length + int((range_start - self.depth_bound[0]) // self.depth_bound[2])] \
                         = depth_score_single * ref_range_score[:, range_idx:range_idx + 1]
            del depth_score_single, mu_repeat, ref_sigma
        return stereo_depth

    @auto_fp16()
    def forward(self, ref_dict, src_dict, trans_dict, img_metas, return_depth=False):
        """Forward function for StereoViewTrans.
        """
        # calculate stereo depth score
        stereo_depth = self.forward_stereo(ref_dict, src_dict, trans_dict, img_metas)
        depth_score = ref_dict['depth'] + self.downsample_net(stereo_depth)
        depth_score = depth_score.softmax(1)
        # calculate splited image feature
        ref_context = ref_dict['context']
        feat_with_depth = depth_score.unsqueeze(1) * ref_context.unsqueeze(2)
        feat_with_depth = feat_with_depth.reshape(-1, self.num_cams, *feat_with_depth.shape[1:])
        # shape: (N, Cam, Depth, H, W, C)
        feat_with_depth = feat_with_depth.permute(0,1,3,4,5,2).contiguous()
        if self.frustum is None:
            self.frustum = self.create_frustum(img_metas, device=ref_context.device)

        sensor2img_key = trans_dict['sensor2img_key']
        frustum = self.frustum.clone()
        frustum = frustum.repeat(*sensor2img_key.shape[:2],1,1,1,1)

        # undo post-transformation for image
        if 'img_aug_mat_key' in trans_dict:
            img_aug_mat_key = trans_dict['img_aug_mat_key'].float()
            img_aug_mat_key = img_aug_mat_key.inverse().to(frustum)
            img_aug_mat_key = img_aug_mat_key.permute(0,2,1)
            img_aug_mat_key = img_aug_mat_key.reshape(*sensor2img_key.shape[:2], 1, 1, 4, 4)
            frustum = frustum @ img_aug_mat_key
            
        # project cam to aligned LiDAR
        img_coord = torch.cat([frustum[...,:2] * frustum[...,2:3], 
                               frustum[...,2:]], dim=-1)
        img2sensor_key = sensor2img_key.inverse()[:,:,None,None].to(img_coord)
        sensor_coord = img_coord @ img2sensor_key.permute(0,1,2,3,5,4)
        sensor_coord = sensor_coord[...,:3]
        # undo post-transformation for BEV here
        if 'bev_aug_mat' in trans_dict:
            bev_aug_mat = trans_dict['bev_aug_mat']
            bev_aug_mat = bev_aug_mat.reshape(-1,1,1,1,3,3)
            sensor_coord = sensor_coord @ bev_aug_mat

        range_start = self.pc_range[:3].reshape(1,1,1,1,1,-1).to(device=sensor_coord.device)
        range_size = self.voxel_size.reshape(1,1,1,1,1,-1).to(device=sensor_coord.device)
        geom_coord = (sensor_coord - range_start) / range_size

        voxel_space = voxel_pooling(geom_coord.int(),
                                    feat_with_depth.float(),
                                    self.voxel_shape.to(geom_coord.device))
        del feat_with_depth, img_coord, sensor_coord, geom_coord, frustum
        
        if return_depth:
            return voxel_space, depth_score
        else:
            return voxel_space