# Copyright (c) 2023, NVIDIA Corporation & Affiliates. All rights reserved. 
# 
# This work is made available under the Nvidia Source Code License-NC. 
# To view a copy of this license, visit 
# https://github.com/NVlabs/DQTrack/blob/main/LICENSE
# Modified from mmDetection3D (https://github.com/open-mmlab/mmdetection3d)
# Copyright (c) OpenMMLab. All rights reserved.
import numpy as np

from mmdet3d.datasets.builder import DATASETS


@DATASETS.register_module()
class NonOverlapDataset(object):
    """A wrapper of class sampled dataset with ann_file path. Implementation of
    non-overlap dataset sampling.

    Args:
        dataset (:obj:`CustomDataset`): The dataset to be class sampled.
    """

    def __init__(self, dataset, clip_double=False):
        self.dataset = dataset
        self.CLASSES = dataset.CLASSES
        self.clip_double = clip_double
        self.cat2id = {name: i for i, name in enumerate(self.CLASSES)}
        self.sample_indices = self._get_sample_indices()
        # self.dataset.data_infos = self.data_infos
        if hasattr(self.dataset, 'flag'):
            self.flag = np.array(
                [self.dataset.flag[ind] for ind in self.sample_indices],
                dtype=np.uint8)

    def _get_sample_indices(self):
        """Load annotations from ann_file.

        Args:
            ann_file (str): Path of the annotation file.

        Returns:
            list[dict]: List of annotations after class sampling.
        """
        sampled_num = self.dataset.num_frames_per_sample
        sample_indices = np.arange(len(self.dataset))
        # sample from the start of the clip
        convert_indices = sample_indices[::sampled_num]
        # sample from the end of the clip
        if self.clip_double:
            double_indices = sample_indices[sampled_num-1::sampled_num]
            convert_indices = np.concatenate([convert_indices, double_indices])

        return convert_indices

    def __getitem__(self, idx):
        """Get item from infos according to the given index.

        Returns:
            dict: Data dictionary of the corresponding index.
        """
        ori_idx = self.sample_indices[idx]
        return self.dataset[ori_idx]

    def __len__(self):
        """Return the length of data infos.

        Returns:
            int: Length of data infos.
        """
        return len(self.sample_indices)
