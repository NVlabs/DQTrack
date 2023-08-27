# Copyright (c) 2023, NVIDIA Corporation & Affiliates. All rights reserved. 
# 
# This work is made available under the Nvidia Source Code License-NC. 
# To view a copy of this license, visit 
# https://github.com/NVlabs/DQTrack/blob/main/LICENSE
# Modified from UVTR (https://github.com/dvlab-research/UVTR)
from re import I
import mmcv
import torch
import numpy as np

from mmdet.datasets.builder import PIPELINES

@PIPELINES.register_module()
class LoadMultiViewMultiSweepImageFromFiles(object):
    """Load multi channel images from a list of separate channel files.

    Expects results['img_filename'] to be a list of filenames.

    Args:
        to_float32 (bool): Whether to convert the img to float32.
            Defaults to False.
        color_type (str): Color type of the file. Defaults to 'unchanged'.
    """

    def __init__(self, to_float32=False, sweep_num=1, random_sweep=False, load_depth=False, color_type='unchanged'):
        self.to_float32 = to_float32
        self.color_type = color_type
        self.sweep_num = sweep_num
        self.random_sweep = random_sweep
        self.load_depth = load_depth

    def __call__(self, results):
        """Call function to load multi-view image from files.

        Args:
            results (dict): Result dict containing multi-view image filenames.

        Returns:
            dict: The result dict containing the multi-view image data. \
                Added keys and values are described below.

                - filename (str): Multi-view image filenames.
                - img (np.ndarray): Multi-view image arrays.
                - img_shape (tuple[int]): Shape of multi-view image arrays.
                - ori_shape (tuple[int]): Shape of original image arrays.
                - pad_shape (tuple[int]): Shape of padded image arrays.
                - scale_factor (float): Scale factor.
                - img_norm_cfg (dict): Normalization configuration of images.
        """
        filename = results['img_filename']
        results['filename'] = filename
        # img is of shape (h, w, c, num_views)
        img = np.stack(
            [mmcv.imread(name, self.color_type) for name in filename], axis=-1)
        # load depth map
        if self.load_depth:
            point_depth = []
            for _name in results['depth_filename']:
                _depth = np.fromfile(_name, dtype=np.float32, count=-1).reshape(-1, 3)
                point_depth.append(_depth)
            results['point_depth'] = point_depth
        
        img_sweeps = []
        sweeps_paths = results['cam_sweeps_paths']
        sweeps_ids = results['cam_sweeps_id']
        sweeps_time = results['cam_sweeps_time']
        if self.random_sweep:
            random_num = np.random.randint(0, self.sweep_num)
            sweeps_paths = [_sweep[:random_num] for _sweep in sweeps_paths]
            sweeps_ids = [_sweep[:random_num] for _sweep in sweeps_ids]
        else:
            random_num = self.sweep_num

        for _idx in range(len(sweeps_paths[0])):
            _sweep = np.stack(
                [mmcv.imread(name_list[_idx], self.color_type) for name_list in sweeps_paths], axis=-1)
            img_sweeps.append(_sweep)

        # add img sweeps to raw image
        img = np.stack([img, *img_sweeps], axis=-1)
        # img is of shape (h, w, c, num_views * sweep_num)
        img = img.reshape(*img.shape[:-2], -1)

        if self.to_float32:
            img = img.astype(np.float32)

        results['sweeps_paths'] = [[filename[_idx]] + sweeps_paths[_idx] for _idx in range(len(filename))]
        results['sweeps_ids'] = np.stack([[0]+_id for _id in sweeps_ids], axis=-1)
        results['sweeps_time'] = np.stack([[0]+_time for _time in sweeps_time], axis=-1)
        # unravel to list, see `DefaultFormatBundle` in formating.py
        # which will transpose each image separately and then stack into array
        results['img'] = [img[..., i] for i in range(img.shape[-1])]
        results['img_shape'] = img.shape
        results['ori_shape'] = img.shape
        # Set initial values for default meta_keys
        results['pad_shape'] = img.shape
        results['scale_factor'] = 1.0
        num_channels = 1 if len(img.shape) < 3 else img.shape[2]
        results['img_norm_cfg'] = dict(
            mean=np.zeros(num_channels, dtype=np.float32),
            std=np.ones(num_channels, dtype=np.float32),
            to_rgb=False)

        # add sweep matrix to raw matrix
        results['lidar2img'] = [np.stack([results['lidar2img'][_idx], 
                                         *results['lidar2img_sweeps'][_idx][:random_num]], axis=0) 
                                         for _idx in range(len(results['lidar2img']))]
        results['lidar2cam'] = [np.stack([results['lidar2cam'][_idx], 
                                         *results['lidar2cam_sweeps'][_idx][:random_num]], axis=0) 
                                         for _idx in range(len(results['lidar2cam']))]
        results['cam_intrinsic'] = [np.stack([results['cam_intrinsic'][_idx], 
                                         *results['cam_sweeps_intrinsics'][_idx][:random_num]], axis=0) 
                                         for _idx in range(len(results['cam_intrinsic']))]
        results.pop('lidar2img_sweeps')
        results.pop('lidar2cam_sweeps')
        results.pop('cam_sweeps_intrinsics')

        return results

    def __repr__(self):
        """str: Return a string that describes the module."""
        repr_str = self.__class__.__name__
        repr_str += f'(to_float32={self.to_float32}, '
        repr_str += f"color_type='{self.color_type}')"
        return repr_str


@PIPELINES.register_module()
class GenerateDepthFromPoints(object):
    """Generate depth map from point clouds.
    """
    def __init__(self, view_num=6):
        self.view_num = view_num

    def __call__(self, results):
        """Call function to generate depth map.

        Args:
            results (dict): Result dict containing multi-view image and depth map.
        """
        depth_lst = []
        for _idx in range(self.view_num):
            depth_map = np.zeros(results['img_shape'][:2])
            cam_depth = results['point_depth'][_idx]
            depth_coords = cam_depth[:, :2].astype(np.int16)
            valid_mask = ((depth_coords[:, 1] < results['img_shape'][0])
                  & (depth_coords[:, 0] < results['img_shape'][1])
                  & (depth_coords[:, 1] >= 0)
                  & (depth_coords[:, 0] >= 0))
            depth_map[depth_coords[valid_mask, 1],
                      depth_coords[valid_mask, 0]] = cam_depth[valid_mask, 2]
            depth_lst.append(torch.Tensor(depth_map))
        
        results['depth_map'] = depth_lst
        return results

@PIPELINES.register_module()
class LoadMultiViewImageFromMultiSweepsFiles(object):
    """Load multi channel images from a list of separate channel files.
    Expects results['img_filename'] to be a list of filenames.
    Args:
        to_float32 (bool): Whether to convert the img to float32.
            Defaults to False.
        color_type (str): Color type of the file. Defaults to 'unchanged'.
    """

    def __init__(self, 
                sweeps_num=5,
                to_float32=False, 
                file_client_args=dict(backend='disk'),
                pad_empty_sweeps=False,
                sweep_range=[3,27],
                sweeps_id = None,
                color_type='unchanged',
                sensors = ['CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_FRONT_LEFT', 'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT'],
                test_mode=True,
                prob=1.0,
                ):

        self.sweeps_num = sweeps_num    
        self.to_float32 = to_float32
        self.color_type = color_type
        self.file_client_args = file_client_args.copy()
        self.file_client = None
        self.pad_empty_sweeps = pad_empty_sweeps
        self.sensors = sensors
        self.test_mode = test_mode
        self.sweeps_id = sweeps_id
        self.sweep_range = sweep_range
        self.prob = prob
        if self.sweeps_id:
            assert len(self.sweeps_id) == self.sweeps_num

    def __call__(self, results):
        """Call function to load multi-view image from files.
        Args:
            results (dict): Result dict containing multi-view image filenames.
        Returns:
            dict: The result dict containing the multi-view image data. \
                Added keys and values are described below.
                - filename (str): Multi-view image filenames.
                - img (np.ndarray): Multi-view image arrays.
                - img_shape (tuple[int]): Shape of multi-view image arrays.
                - ori_shape (tuple[int]): Shape of original image arrays.
                - pad_shape (tuple[int]): Shape of padded image arrays.
                - scale_factor (float): Scale factor.
                - img_norm_cfg (dict): Normalization configuration of images.
        """
        sweep_imgs_list = []
        timestamp_imgs_list = []
        imgs = results['img']
        img_timestamp = results['img_timestamp']
        lidar_timestamp = results['timestamp']
        results['lidar_timestamp'] = lidar_timestamp
        img_timestamp = [lidar_timestamp - timestamp for timestamp in img_timestamp]
        sweep_imgs_list.extend(imgs)
        timestamp_imgs_list.extend(img_timestamp)
        nums = len(imgs)

        if self.pad_empty_sweeps and len(results['sweeps']) == 0:
            for i in range(self.sweeps_num):
                sweep_imgs_list.extend(imgs)
                mean_time = (self.sweep_range[0] + self.sweep_range[1]) / 2.0 * 0.083
                timestamp_imgs_list.extend([time + mean_time for time in img_timestamp])
                for j in range(nums):
                    results['filename'].append(results['filename'][j])
                    results['lidar2img'].append(np.copy(results['lidar2img'][j]))
                    results['intrinsics'].append(np.copy(results['intrinsics'][j]))
                    results['extrinsics'].append(np.copy(results['extrinsics'][j]))
        else:
            if self.sweeps_id:
                choices = self.sweeps_id
            elif len(results['sweeps']) <= self.sweeps_num:
                choices = np.arange(len(results['sweeps']))
            elif self.test_mode:
                choices = [int((self.sweep_range[0] + self.sweep_range[1])/2) - 1] 
            else:
                if np.random.random() < self.prob:
                    if self.sweep_range[0] < len(results['sweeps']):
                        sweep_range = list(range(self.sweep_range[0], min(self.sweep_range[1], len(results['sweeps']))))
                    else:
                        sweep_range = list(range(self.sweep_range[0], self.sweep_range[1]))
                    choices = np.random.choice(sweep_range, self.sweeps_num, replace=False)
                else:
                    choices = [int((self.sweep_range[0] + self.sweep_range[1])/2) - 1] 
            
            for idx in choices:
                sweep_idx = min(idx, len(results['sweeps']) - 1)
                sweep = results['sweeps'][sweep_idx]
                if len(sweep.keys()) < len(self.sensors):
                    sweep = results['sweeps'][sweep_idx - 1]
                results['filename'].extend([sweep[sensor]['data_path'] for sensor in self.sensors])

                img = np.stack([mmcv.imread(sweep[sensor]['data_path'], self.color_type) for sensor in self.sensors], axis=-1)
                
                if self.to_float32:
                    img = img.astype(np.float32)
                img = [img[..., i] for i in range(img.shape[-1])]
                sweep_imgs_list.extend(img)
                sweep_ts = [lidar_timestamp - sweep[sensor]['timestamp'] / 1e6  for sensor in self.sensors]
                timestamp_imgs_list.extend(sweep_ts)
                for sensor in self.sensors:
                    results['lidar2img'].append(sweep[sensor]['lidar2img'])
                    results['intrinsics'].append(sweep[sensor]['intrinsics'])
                    results['extrinsics'].append(sweep[sensor]['extrinsics'])
        results['img'] = sweep_imgs_list
        results['timestamp'] = timestamp_imgs_list

        return results