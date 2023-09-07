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

from torch import Tensor

__all__ = [
    "compute_frame_resolution",
    "tensor_to_tiled",
    "tiled_to_tensor",
    "tensor_to_quilted",
    "quilted_to_Tensor",
]


def compute_frame_resolution(num_channels, channel_height, channel_width):
    short_edge = int(math.sqrt(num_channels))

    while (num_channels % short_edge) != 0:
        short_edge -= 1

    long_edge = num_channels // short_edge

    assert (short_edge * long_edge) == num_channels

    # try to make it close to a square
    if channel_height > channel_width:
        height = short_edge * channel_height
        width = long_edge * channel_width
    else:
        width = short_edge * channel_width
        height = long_edge * channel_height

    return (height, width)


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


def tensor_to_quilted(x: Tensor, num_samples_in_width, num_samples_in_height):
    raise NotImplementedError


def quilted_to_Tensor():
    raise NotImplementedError
