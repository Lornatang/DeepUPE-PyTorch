# Copyright 2023 Dakewe Biotech Corporation. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# wIThOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F_torch
from torch import Tensor

__all__ = [
    "DeepUPE",
    "COLORLoss", "TVLoss",
    "deep_upe",
]


class DeepUPE(nn.Module):

    def __init__(
            self,
            in_channels: int = 3,
            out_channels: int = 4,
            luma_bins: int = 8,
            channel_multiplier: int = 1,
            spatial_bin: int = 8,
            batch_norm: bool = True,
            low_resolution_size: int = 256,
    ) -> None:
        super(DeepUPE, self).__init__()
        self.low_resolution_size = low_resolution_size

        self.coefficients = Coefficients(in_channels,
                                         out_channels,
                                         luma_bins,
                                         channel_multiplier,
                                         spatial_bin,
                                         batch_norm,
                                         low_resolution_size)
        self.get_feature_map = GuideNN(batch_norm)
        self.slice = Slice()
        self.apply_coefficients = ApplyCoefficients()

    def forward(self, x: Tensor) -> Tensor:
        x1 = F_torch.interpolate(x, size=[self.low_resolution_size, self.low_resolution_size])
        x2 = x

        coefficients = self.coefficients(x1)
        feature_map = self.get_feature_map(x2)
        slice_coefficients = self.slice(coefficients, feature_map)
        out = self.apply_coefficients(slice_coefficients, x2)

        return out


class BasicConvBlock(nn.Module):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            kernel_size: int = 3,
            stride: int = 1,
            padding: int = 1,
            activation: nn.ReLU | Any = nn.ReLU,
            batch_norm: bool = False,
    ) -> None:
        super(BasicConvBlock, self).__init__()
        bias = False if batch_norm else True

        self.conv = nn.Conv2d(in_channels,
                              out_channels,
                              kernel_size,
                              stride,
                              padding,
                              bias=bias)
        self.batch_norm = nn.BatchNorm2d(out_channels) if batch_norm else None
        self.activation = activation() if activation else None

    def forward(self, x: Tensor) -> Tensor:
        x = self.conv(x)
        if self.batch_norm:
            x = self.batch_norm(x)
        if self.activation:
            x = self.activation(x)

        return x


class FullyConnect(nn.Module):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            activation: nn.ReLU | Any = nn.ReLU,
            batch_norm: bool = False,
    ) -> None:
        super(FullyConnect, self).__init__()
        self.fc = nn.Linear(in_channels, out_channels)
        self.batch_norm = nn.BatchNorm1d(out_channels) if batch_norm else None
        self.activation = activation() if activation else None

    def forward(self, x: Tensor) -> Tensor:
        x = self.fc(x)

        if self.batch_norm:
            x = self.batch_norm(x)
        if self.activation:
            x = self.activation(x)

        return x


class Slice(nn.Module):
    def __init__(self) -> None:
        super(Slice, self).__init__()

    def forward(self, bilateral_grid: Tensor, feature_map: Tensor) -> Tensor:
        """Slice the bilateral grid.

        Args:
            bilateral_grid (Tensor): The bilateral grid.
            feature_map (Tensor): The feature map.

        Returns:
            coefficient (Tensor): The slice bilateral grid.
        """

        batch_size, _, h, w = feature_map.shape

        # Creates grids of coordinates specified by the 1D inputs
        hg, wg = torch.meshgrid([torch.arange(0, h), torch.arange(0, w)], indexing="ij")  # [0,511] hxw
        hg = hg.to(bilateral_grid.device)
        wg = wg.to(bilateral_grid.device)
        hg = hg.float().repeat(batch_size, 1, 1).unsqueeze(3) / (h - 1) * 2 - 1  # norm to [-1,1]
        wg = wg.float().repeat(batch_size, 1, 1).unsqueeze(3) / (w - 1) * 2 - 1  # norm to [-1,1]

        # Concatenate the grid and the feature map
        feature_map = feature_map.permute(0, 2, 3, 1).contiguous()
        feature = torch.cat([wg, hg, feature_map], dim=3).unsqueeze(1)  # Nx1xHxWx3
        coefficient = F_torch.grid_sample(bilateral_grid, feature, padding_mode="border", align_corners=True)
        coefficient = coefficient.squeeze(2)

        return coefficient


class Coefficients(nn.Module):

    def __init__(
            self,
            in_channels: int = 3,
            out_channels: int = 4,
            luma_bins: int = 8,
            channel_multiplier: int = 1,
            spatial_bin: int = 8,
            batch_norm: bool = True,
            low_resolution_size: int = 256,
    ) -> None:
        super(Coefficients, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.luma_bins = luma_bins
        self.channel_multiplier = channel_multiplier
        self.spatial_bin = spatial_bin

        # splat features
        num_layers_splat = int(np.log2(low_resolution_size / spatial_bin))

        self.splat_features = nn.ModuleList()
        prev_channels = in_channels
        splat_channels = int(channel_multiplier * luma_bins)
        for i in range(num_layers_splat):
            use_bn = batch_norm if i > 0 else False
            self.splat_features.append(BasicConvBlock(prev_channels,
                                                      int(channel_multiplier * (2 ** i) * luma_bins),
                                                      3,
                                                      2,
                                                      1,
                                                      nn.ReLU,
                                                      use_bn))
            splat_channels = channel_multiplier * (2 ** i) * luma_bins
            prev_channels = splat_channels

        # global features
        num_layers_global = int(np.log2(spatial_bin / 4))
        self.global_features_conv = nn.ModuleList()
        self.global_features_fc = nn.ModuleList()
        for i in range(num_layers_global):
            self.global_features_conv.append(BasicConvBlock(prev_channels,
                                                            int(channel_multiplier * 8 * luma_bins),
                                                            3,
                                                            2,
                                                            1,
                                                            nn.ReLU,
                                                            batch_norm))
            prev_channels = int(channel_multiplier * 8 * luma_bins)

        total_layers = num_layers_splat + num_layers_global
        prev_channels = int(prev_channels * (low_resolution_size / 2 ** total_layers) ** 2)
        self.global_features_fc.append(FullyConnect(prev_channels,
                                                    int(32 * channel_multiplier * luma_bins),
                                                    nn.ReLU,
                                                    batch_norm))
        self.global_features_fc.append(FullyConnect(int(32 * channel_multiplier * luma_bins),
                                                    int(16 * channel_multiplier * luma_bins),
                                                    nn.ReLU,
                                                    batch_norm))
        self.global_features_fc.append(FullyConnect(int(16 * channel_multiplier * luma_bins),
                                                    int(8 * channel_multiplier * luma_bins),
                                                    None,
                                                    batch_norm))

        # local features
        self.local_features = nn.ModuleList()
        self.local_features.append(BasicConvBlock(splat_channels,
                                                  int(8 * channel_multiplier * luma_bins),
                                                  3,
                                                  1,
                                                  1,
                                                  nn.ReLU,
                                                  batch_norm))
        self.local_features.append(BasicConvBlock(int(8 * channel_multiplier * luma_bins),
                                                  int(8 * channel_multiplier * luma_bins),
                                                  3,
                                                  1,
                                                  1,
                                                  None,
                                                  False))

        # prediction
        self.conv_out = BasicConvBlock(int(8 * channel_multiplier * luma_bins),
                                       int(luma_bins * out_channels * in_channels),
                                       1,
                                       1,
                                       0,
                                       None,
                                       False)
        self.relu = nn.ReLU()

    def forward(self, low_resolution_input: Tensor) -> Tensor:
        batch_size = low_resolution_input.shape[0]

        x = low_resolution_input
        for layer in self.splat_features:
            x = layer(x)
        splat_features = x

        for layer in self.global_features_conv:
            x = layer(x)
        x = x.view(batch_size, -1)

        for layer in self.global_features_fc:
            x = layer(x)
        global_features = x

        x = splat_features
        for layer in self.local_features:
            x = layer(x)
        local_features = x

        fusion_grid = local_features
        fusion_global = global_features.view(batch_size, int(8 * self.channel_multiplier * self.luma_bins), 1, 1)
        fusion = self.relu(fusion_grid + fusion_global)

        x = self.conv_out(fusion)

        # B x Coefficients x Luma x Spatial x Spatial
        x = x.view(batch_size, self.in_channels * self.out_channels, self.luma_bins, self.spatial_bin, self.spatial_bin)

        return x


class ApplyCoefficients(nn.Module):
    def __init__(self):
        super(ApplyCoefficients, self).__init__()

    def forward(self, coefficient: Tensor, full_res_input: Tensor) -> Tensor:
        R = torch.sum(full_res_input * coefficient[:, 0:3, :, :], dim=1, keepdim=True) + coefficient[:, 3:4, :, :]
        G = torch.sum(full_res_input * coefficient[:, 4:7, :, :], dim=1, keepdim=True) + coefficient[:, 7:8, :, :]
        B = torch.sum(full_res_input * coefficient[:, 8:11, :, :], dim=1, keepdim=True) + coefficient[:, 11:12, :, :]

        out = torch.cat([R, G, B], dim=1)

        return out


class GuideNN(nn.Module):
    def __init__(self, batch_norm: bool) -> None:
        super(GuideNN, self).__init__()
        self.conv1 = BasicConvBlock(3, 16, 1, 1, 0, nn.ReLU, batch_norm)
        self.conv2 = BasicConvBlock(16, 1, 1, 1, 0, nn.Tanh, True)

    def forward(self, x: Tensor) -> Tensor:
        x = self.conv1(x)
        x = self.conv2(x)

        return x


class COLORLoss(nn.Module):
    def __init__(self) -> None:
        super(COLORLoss, self).__init__()

    def forward(self, pred_tensor: Tensor, gt_tensor: Tensor) -> Tensor:
        assert pred_tensor.size() == gt_tensor.size(), "Two tensor must have the same size"

        batch_size, channels, height, width = pred_tensor.shape

        pred_reflect_view = pred_tensor.view(batch_size, channels, height * width).permute(0, 2, 1)
        gt_reflect_view = gt_tensor.view(batch_size, channels, height * width).permute(0, 2, 1)
        pred_reflect_norm = F_torch.normalize(pred_reflect_view, dim=-1)
        gt_reflect_norm = F_torch.normalize(gt_reflect_view, dim=-1)
        cose_value = pred_reflect_norm * gt_reflect_norm
        cose_value = torch.sum(cose_value, dim=-1)
        color_loss = torch.mean(1 - cose_value)

        return color_loss


class TVLoss(nn.Module):
    def __init__(self, alpha: float = 1.2, lamda: float = 1.5) -> None:
        super(TVLoss, self).__init__()
        self.alpha = alpha
        self.lamda = lamda

    def forward(self, pred_tensor: Tensor, gt_tensor: Tensor, pred_illmunination=None, image=None) -> Tensor:
        assert pred_tensor.size() == gt_tensor.size(), "Two tensor must have the same size"

        L = torch.log(image + 1e-4)
        dx = L[:, :, 1:, :-1] - L[:, :, 1:, 1:]
        dy = L[:, :, :-1, 1:] - L[:, :, 1:, 1:]

        dx = self.lamda / (torch.pow(torch.abs(dx), self.alpha) + 1e-4)
        dy = self.lamda / (torch.pow(torch.abs(dy), self.alpha) + 1e-4)
        S = pred_illmunination
        x_loss = dx * torch.pow(S[:, :, 1:, :-1] - S[:, :, 1:, 1:], 2)
        y_loss = dy * torch.pow(S[:, :, :-1, 1:] - S[:, :, 1:, 1:], 2)
        tv_loss = torch.mean(x_loss + y_loss)

        return tv_loss


def deep_upe(**kwargs) -> DeepUPE:
    model = DeepUPE(in_channels=3,
                    out_channels=4,
                    luma_bins=8,
                    channel_multiplier=1,
                    spatial_bin=8,
                    batch_norm=True,
                    **kwargs)

    return model
