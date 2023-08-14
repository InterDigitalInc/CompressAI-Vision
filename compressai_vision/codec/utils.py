# Copyright (c) 2022-2023, InterDigital Communications, Inc
# All rights reserved.

# Redistribution and use in source and binary forms, with or without
# modification, are permitted (subject to the limitations in the disclaimer
# below) provided that the following conditions are met:

# * Redistributions of source code must retain the above copyright notice,
#   this list of conditions and the following disclaimer.
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
# * Neither the name of InterDigital Communications, Inc nor the names of its
#   contributors may be used to endorse or promote products derived from this
#   software without specific prior written permission.

# NO EXPRESS OR IMPLIED LICENSES TO ANY PARTY'S PATENT RIGHTS ARE GRANTED BY
# THIS LICENSE. THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND
# CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT
# NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
# PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS;
# OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
# WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR
# OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF
# ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import logging
import math
from typing import Dict

import torch
from torch import Tensor

MIN_MAX_DATASET = {
    "mpeg-oiv6-detection": (-26.426828384399414, 28.397470474243164),
    "TVD": (-4.722218990325928, 48.58344268798828),
    "HiEve": (-1.0795, 11.8232),
    "SFU": (-17.8848, 16.69417),
}


def tensor_to_tiled(x: Tensor, tiled_frame_resolution):
    assert x.dim() == 4 and isinstance(x, Tensor)
    _, _, H, W = x.size()

    num_channels_in_height = tiled_frame_resolution[0] // H
    num_channels_in_width = tiled_frame_resolution[1] // W

    A = x.reshape(num_channels_in_height, num_channels_in_width, H, W)
    B = A.swapaxes(1, 2)
    tiled = B.reshape(tiled_frame_resolution[0], tiled_frame_resolution[1])

    return tiled


def tiled_to_tensor(x: Tensor, channel_resolution):
    assert x.dim() == 2 and isinstance(x, Tensor)
    frmH, frmW = x.size()

    num_channels_in_height = frmH // channel_resolution[0]
    num_channels_in_width = frmW // channel_resolution[1]
    total_num_channels = int(num_channels_in_height * num_channels_in_width)

    A = x.reshape(
        num_channels_in_height,
        channel_resolution[0],
        num_channels_in_width,
        channel_resolution[1],
    )
    B = A.swapaxes(1, 2)
    feature_tensor = B.reshape(
        1, total_num_channels, channel_resolution[0], channel_resolution[1]
    )

    return feature_tensor


def min_max_normalization(x, min: float, max: float, bitdepth: int = 10):
    max_num_bins = (2**bitdepth) - 1
    out = ((x - min) / (max - min)).clamp_(0, 1)
    mid_level = -min / (max - min)
    return (out * max_num_bins).floor(), int(mid_level * max_num_bins + 0.5)


def min_max_inv_normalization(x, min: float, max: float, bitdepth: int = 10):
    out = x / ((2**bitdepth) - 1)
    out = (out * (max - min)) + min
    return out


def feature_pyramid_to_frame(x: Dict[str, Tensor], packing_all_in_one=False):
    """rehape the feature pyramid to a frame
    This function is specific to detectron2 for now
    """
    # 'p2' is the base for the size of to-be-formed frame
    _, C, H, W = x["p2"].size()

    _, fixedW = compute_frame_resolution(C, H, W)

    tiled_frame = {}

    feature_size = {}

    subframe_heights = {}
    for key, tensor in x.items():
        N, C, H, W = tensor.size()

        assert N == 1, f"the batch size shall be one, but got {N}"
        frmH, frmW = compute_frame_resolution(C, H, W)
        rescale = fixedW // frmW if packing_all_in_one else 1

        new_frmH = frmH // rescale
        new_frmW = frmW * rescale

        frame = tensor_to_tiled(tensor, (new_frmH, new_frmW))
        tiled_frame.update({key: frame})
        feature_size.update({key: tensor.size()})
        subframe_heights.update({key: new_frmH})

    if packing_all_in_one:
        for key, subframe in tiled_frame.items():
            if key == "p2":
                out = subframe
            else:
                out = torch.cat([out, subframe], dim=0)
        tiled_frame = out

    return tiled_frame, feature_size, subframe_heights


def frame_to_feature_pyramid(
    x, tensor_shape: Dict, subframe_height: Dict, packing_all_in_one=False
):
    """reshape a frame of channels into the feature pyramid"""

    assert isinstance(x, (Tensor, Dict))
    top_y = 0
    tiled_frame = {}

    if packing_all_in_one:
        for key, height in subframe_height.items():
            tiled_frame.update({key: x[top_y : top_y + height, :]})
            top_y = top_y + height
    else:
        assert isinstance(x, Dict)
        tiled_frame = x
    feature_tensor = {}

    for key, frame in tiled_frame.items():
        _, numChs, chH, chW = tensor_shape[key]
        tensor = tiled_to_tensor(frame, (chH, chW))
        assert tensor.size(1) == numChs
        feature_tensor.update({key: tensor})
    return feature_tensor
