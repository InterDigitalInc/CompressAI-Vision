# Copyright (c) 2022-2024, InterDigital Communications, Inc
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

import json
import math
from typing import Dict

import torch
import torch.nn.functional as F
from torch import Tensor

MIN_MAX_DATASET = {
    "mpeg-oiv6-detection": (
        -26.426828384399414,
        28.397470474243164,
    ),  # According to the anchor scripts -> global_max = 20.246625900268555, global_min = -23.09193229675293
    "mpeg-oiv6-segmentation": (-26.426828384399414, 28.397470474243164),
    "MPEGTVDTRACKING": (-4.722218990325928, 48.58344268798828),
    "MPEGHIEVE": (-1.0795, 11.8232),
    "SFUHW": (-17.8848, 16.69417),
}


def min_max_normalization(x, min: float, max: float, bitdepth: int = 10):
    max_num_bins = (2**bitdepth) - 1
    out = ((x - min) / (max - min)).clamp_(0, 1)
    mid_level = -min / (max - min)
    return (out * max_num_bins).floor(), int(mid_level * max_num_bins + 0.5)


def min_max_inv_normalization(x, min: float, max: float, bitdepth: int = 10):
    out = x / ((2**bitdepth) - 1)
    out = (out * (max - min)) + min
    return out


def pad(x, p=2**6, bottom_right=False):
    h, w = x.size(2), x.size(3)
    H = (h + p - 1) // p * p
    W = (w + p - 1) // p * p

    if bottom_right is True:
        padding_left = 0
        padding_top = 0
    else:
        padding_left = (W - w) // 2
        padding_top = (H - h) // 2

    padding_right = W - w - padding_left
    padding_bottom = H - h - padding_top
    return F.pad(
        x,
        (padding_left, padding_right, padding_top, padding_bottom),
        mode="constant",
        value=0,
    )


def crop(x, size, bottom_right=False):
    H, W = x.size(2), x.size(3)
    h, w = size

    if bottom_right is True:
        padding_left = 0
        padding_top = 0
    else:
        padding_left = (W - w) // 2
        padding_top = (H - h) // 2

    padding_right = W - w - padding_left
    padding_bottom = H - h - padding_top

    return F.pad(
        x,
        (-padding_left, -padding_right, -padding_top, -padding_bottom),
        mode="constant",
        value=0,
    )


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


class FpnUtils:
    """Utilities for feature pyramid networks (FPN)."""

    def dump_fpn_sizes_json(self, file_prefix, bitstream_name, codec_output_dir):
        """
        Dump the FPN sizes JSON file.
        This function dumps the FPN sizes JSON file for a given split model.

        Args:
        - file_prefix (str): The file prefix to be used for the JSON file. If empty, it uses the bitstream name.
        - bitstream_name (str): The name of the bitstream.
        - codec_output_dir (Path): The directory where the codec output is located.

        Raises:
        - SystemExit: This function is just meant to be used once to dump file and exit.

        Returns:
        - None
        """
        filename = file_prefix if file_prefix != "" else bitstream_name.split("_qp")[0]
        fpn_sizes_json = codec_output_dir / f"{filename}.json"
        with fpn_sizes_json.open("wb") as f:
            output = {
                "fpn": self.feature_size,
                "subframe_heights": self.subframe_heights,
            }
            f.write(json.dumps(output, indent=4).encode())
        print(f"fpn sizes json dump generated, exiting")
        raise SystemExit(0)

    def reshape_feature_pyramid_to_frame(self, x: Dict, packing_all_in_one=False):
        """rehape the feature pyramid to a frame"""

        # find the largest tensor
        x_sorted = sorted(
            x.values(), key=lambda item: math.prod(item[1].size()), reverse=True
        )

        nbframes, C, H, W = x_sorted[0].size()
        _, fixedW = compute_frame_resolution(C, H, W)

        assert packing_all_in_one == True, "packing_all_in_one False is not support yet"

        # compute packing subframes
        self.subframe_heights = []
        self.subframe_widths = []
        for i, tensor in enumerate(x_sorted):
            single_tensor = tensor[0:1, ::]
            _, C, H, W = single_tensor.shape

            frmH, frmW = compute_frame_resolution(C, H, W)

            rescale = fixedW // frmW if packing_all_in_one else 1

            new_frmH = frmH // rescale
            new_frmW = frmW * rescale

            self.subframe_heights.append(new_frmH)
            self.subframe_widths.append(new_frmW)

        packed_frame_list = []
        for n in range(nbframes):
            tiles = []
            for i, tensor in enumerate(x_sorted):
                single_tensor = tensor[n : n + 1, ::]
                N, C, H, W = single_tensor.size()

                assert N == 1, f"the batch size shall be one, but got {N}"

                tile = tensor_to_tiled(
                    single_tensor, (self.subframe_heights[i], self.subframe_widths[i])
                )

                tiles.append(tile)

            if packing_all_in_one:
                packed_frame_list.append(torch.cat(tiles))

        packed_frames = torch.stack(packed_frame_list)

        return packed_frames

    def reshape_frame_to_feature_pyramid(
        self, x, tensor_shape: Dict, subframe_height: Dict, packing_all_in_one=False
    ):
        """reshape a frame of channels into the feature pyramid"""

        assert isinstance(x, (Tensor, Dict))
        assert (
            packing_all_in_one is True
        ), "packing_all_in_one = False is not supported yet"

        top_y = 0
        tiled_frames = {}
        if packing_all_in_one:
            for key, height in subframe_height.items():
                tiled_frames[key] = x[:, top_y : top_y + height, :]
                top_y = top_y + height
        else:
            raise NotImplementedError
            assert isinstance(x, Dict)
            tiled_frames = x

        feature_tensor = {}
        for key, frames in tiled_frames.items():
            _, numChs, chH, chW = tensor_shape[key]

            tensors = torch.cat(
                [tiled_to_tensor(frame, (chH, chW)) for frame in frames]
            )
            assert tensors.size(1) == numChs

            feature_tensor[key] = tensors

        return feature_tensor
