# Copyright (c) 2023, NVIDIA Corporation & Affiliates. All rights reserved. 
# 
# This work is made available under the Nvidia Source Code License-NC. 
# To view a copy of this license, visit 
# https://github.com/NVlabs/DQTrack/blob/main/LICENSE
# Modified from MOTR (https://github.com/megvii-research/MOTR/blob/main/models/qim.py)
# ------------------------------------------------------------------------
# Copyright (c) 2021 megvii-model. All Rights Reserved.
# ------------------------------------------------------------------------
import torch
from torch import nn
from torch.nn import functional as F

class QueryInteraction(nn.Module):
    def __init__(self, in_channels, mid_channels, **kwargs):
        super().__init__()
        dropout = kwargs.get('drop_rate', 0.0)
        self.with_att = kwargs.get('with_att', False)
        self.with_pos = kwargs.get('with_pos', False)
        self.self_attn = nn.MultiheadAttention(in_channels, 8, dropout)
        self.norm1 = nn.LayerNorm(in_channels)
        self.dropout = nn.Dropout(dropout)
        self.ffn = FFN(in_channels, mid_channels, dropout)

    def forward(self, track_embedding, pos_embedding=None):
        # add position embedding
        if self.with_pos and pos_embedding is not None:
            q = k = track_embedding + pos_embedding
        else:
            q = k = track_embedding

        tgt = track_embedding
        # attention
        if self.with_att:
            tgt2 = self.self_attn(q[:, None], k[:, None], value=tgt[:, None])[0][:, 0]
            tgt = tgt + self.dropout(tgt2)
            tgt = self.norm1(tgt)

        # ffn
        tgt = self.ffn(tgt)

        return tgt

class QueryInteractionX(nn.Module):
    def __init__(self, in_channels, mid_channels, **kwargs):
        super().__init__()
        dropout = kwargs.get('drop_rate', 0.0)
        self.with_att = kwargs.get('with_att', False)
        self.with_pos = kwargs.get('with_pos', False)
        self.self_attn = nn.MultiheadAttention(in_channels, 8, dropout)
        self.norm1 = nn.LayerNorm(in_channels)
        self.dropout = nn.Dropout(dropout)
        self.ffn = FFN(in_channels, mid_channels, dropout)

    def forward(self, track_embed, obj_embed, pos_embed=None):
        track_num = len(track_embed)
        query_embed = torch.cat([track_embed, obj_embed], dim=0)
        
        # add position embedding
        if self.with_pos and pos_embed is not None:
            pos_embed = torch.cat([pos_embed, pos_embed], dim=0)
            q = k = query_embed + pos_embed
        else:
            q = k = query_embed

        tgt = query_embed.clone()
        # attention
        if self.with_att:
            tgt2 = self.self_attn(q[:, None], k[:, None], value=tgt[:, None])[0][:, 0]
            tgt = tgt + self.dropout(tgt2)
            tgt = self.norm1(tgt)

        # ffn
        tgt = self.ffn(tgt)

        track_embed = tgt[:track_num]
        obj_embed = tgt[track_num:]

        return track_embed, obj_embed

class FFN(nn.Module):
    def __init__(self, d_model, d_ffn, dropout=0):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ffn)
        self.activation = F.relu
        self.dropout1 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ffn, d_model)
        self.dropout2 = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, tgt):
        tgt2 = self.linear2(self.dropout1(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm(tgt)
        return tgt


class ConvBnReLU3D(nn.Module):
    """Implements of 3d convolution + batch normalization + ReLU."""
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        pad: int = 1,
        dilation: int = 1,
    ) -> None:
        """initialization method for convolution3D + batch normalization + relu module
        Args:
            in_channels: input channel number of convolution layer
            out_channels: output channel number of convolution layer
            kernel_size: kernel size of convolution layer
            stride: stride of convolution layer
            pad: pad of convolution layer
            dilation: dilation of convolution layer
        """
        super(ConvBnReLU3D, self).__init__()
        self.conv = nn.Conv3d(in_channels,
                              out_channels,
                              kernel_size,
                              stride=stride,
                              padding=pad,
                              dilation=dilation,
                              bias=False)
        self.bn = nn.BatchNorm3d(out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """forward method"""
        return F.relu(self.bn(self.conv(x)), inplace=True)