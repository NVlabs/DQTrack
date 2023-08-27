# Copyright (c) 2023, NVIDIA Corporation & Affiliates. All rights reserved. 
# 
# This work is made available under the Nvidia Source Code License-NC. 
# To view a copy of this license, visit 
# https://github.com/NVlabs/DQTrack/blob/main/LICENSE
# Modified from UVTR (https://github.com/dvlab-research/UVTR)
import re
import warnings
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import xavier_init, constant_init
from mmcv.cnn.bricks.registry import (ATTENTION,
                                      TRANSFORMER_LAYER_SEQUENCE)
from mmcv.cnn.bricks.transformer import (MultiScaleDeformableAttention,
                                         TransformerLayerSequence,
                                         build_transformer_layer_sequence)
from mmcv.runner.base_module import BaseModule
from mmcv.runner import force_fp32, auto_fp16
from mmdet.models.utils.builder import TRANSFORMER
from . import UniCrossAtten

@TRANSFORMER.register_module()
class Uni3DTrackDETR(BaseModule):
    """
    Implements the UVTR transformer.
    """
    def __init__(self,
                 num_feature_levels=4,
                 num_cams=6,
                 two_stage_num_proposals=300,
                 decoder=None,
                 fp16_enabled=False,
                 **kwargs):
        super(Uni3DTrackDETR, self).__init__(**kwargs)
        self.decoder = build_transformer_layer_sequence(decoder)
        self.embed_dims = self.decoder.embed_dims
        self.num_feature_levels = num_feature_levels
        self.num_cams = num_cams
        self.two_stage_num_proposals = two_stage_num_proposals
        
        self.init_layers()
        if fp16_enabled:
            self.fp16_enabled = fp16_enabled

    def init_layers(self):
        pass

    def init_weights(self):
        """Initialize the transformer weights."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        for m in self.modules():
            if isinstance(m, MultiScaleDeformableAttention) or isinstance(m, UniCrossAtten):
                m.init_weight()

    @auto_fp16(apply_to=('pts_value', 'img_value', 'query_embed', 'ref_points'))
    def forward(self,
                pts_value,
                img_value,
                query_embed,
                ref_points,
                ref_size=None,
                reg_branches=None,
                **kwargs):
        
        assert query_embed is not None
        if img_value is not None:
            bs = img_value.shape[0]
        else:
            bs = pts_value.shape[0]

        query_pos, query = torch.split(query_embed, self.embed_dims, dim=1)
        query_pos = query_pos.unsqueeze(0).expand(bs, -1, -1)
        query = query.unsqueeze(0).expand(bs, -1, -1)
        reference_points = ref_points.unsqueeze(0).expand(bs, -1, -1)

        if ref_size is not None:
            ref_size = ref_size.unsqueeze(0).expand(bs, -1, -1)
        # DO NOT apply inplace sigmoid to reference_points directly!
        init_reference_out = reference_points.sigmoid()

        # decoder
        query = query.permute(1, 0, 2)
        query_pos = query_pos.permute(1, 0, 2)
        value = {'pts_value':pts_value,
                 'img_value':img_value}
        inter_states, inter_references, inter_box_sizes = self.decoder(
            query=query,
            key=None,
            value=value,
            query_pos=query_pos,
            reference_points=reference_points,
            reg_branches=reg_branches,
            ref_size=ref_size,
            **kwargs)

        inter_references_out = inter_references.sigmoid()
        return inter_states, init_reference_out, inter_references_out, inter_box_sizes

@TRANSFORMER_LAYER_SEQUENCE.register_module()
class UniTrackTransformerDecoder(TransformerLayerSequence):
    """
    Implements the decoder in UVTR transformer.
    """
    def __init__(self, *args, return_intermediate=False, **kwargs):
        super(UniTrackTransformerDecoder, self).__init__(*args, **kwargs)
        self.return_intermediate = return_intermediate

    def forward(self,
                query,
                *args,
                reference_points=None,
                reg_branches=None,
                ref_size=None,
                **kwargs):
        """
        Forward function for `UniTrackTransformerDecoder`.
        Args:
            query (Tensor): Input query with shape
                `(num_query, bs, embed_dims)`.
            reference_points (Tensor): The reference
                points of offset. has shape
                (bs, num_query, 4) when as_two_stage,
                otherwise has shape ((bs, num_query, 2).
            reg_branch: (obj:`nn.ModuleList`): Used for
                refining the regression results. Only would
                be passed when with_box_refine is True,
                otherwise would be passed a `None`.
        Returns:
            Tensor: Results with shape [1, num_query, bs, embed_dims] when
                return_intermediate is `False`, otherwise it has shape
                [num_layers, num_query, bs, embed_dims].
        """
        output = query
        intermediate = []
        intermediate_reference_points = []
        intermediate_box_sizes = []
        for lid, layer in enumerate(self.layers):
            output = layer(
                output,
                *args,
                reference_points=reference_points,
                **kwargs)
            output = output.permute(1, 0, 2)
            if reg_branches is not None:
                tmp = reg_branches[lid](output)
                
                assert reference_points.shape[-1] == 3
                # tmp: (cx, cy, w, l, cz, h, rot_sine, rot_cosine, vx, vy)
                new_reference_points = torch.zeros_like(reference_points)
                new_reference_points[..., :2] = tmp[..., :2] + reference_points[..., :2]
                new_reference_points[..., 2:3] = tmp[..., 4:5] + reference_points[..., 2:3]
                reference_points = new_reference_points.detach()

                if ref_size is not None:
                    ref_size[..., :2] += tmp[..., 2:4]
                    ref_size[..., 2:] += tmp[..., 5:6]
                    if lid > 0:
                        ref_size = ref_size.detach()

            output = output.permute(1, 0, 2)
            if self.return_intermediate:
                intermediate.append(output)
                intermediate_reference_points.append(reference_points)
                intermediate_box_sizes.append(ref_size)

        if self.return_intermediate:
            if ref_size is not None:
                return torch.stack(intermediate), \
                    torch.stack(intermediate_reference_points), \
                    torch.stack(intermediate_box_sizes)
            else:
                return torch.stack(intermediate), \
                    torch.stack(intermediate_reference_points), \
                    None

        return output, reference_points, ref_size