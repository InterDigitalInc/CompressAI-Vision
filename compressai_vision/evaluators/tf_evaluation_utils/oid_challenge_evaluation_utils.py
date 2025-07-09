# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================


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


from __future__ import division, print_function

import base64
import zlib

from typing import Dict

import numpy as np

from pycocotools import mask as coco_mask
from torch import Tensor


def to_normalized_box(mask_np):
    """Decodes binary segmentation masks into np.arrays and boxes.

    Args:
      mask_np: np.ndarray of size NxWxH.

    Returns:
      a np.ndarray of the size Nx4, each row containing normalized coordinates
      [YMin, XMin, YMax, XMax] of a box computed of axis parallel enclosing box of
      a mask.
    """
    coord1, coord2 = np.nonzero(mask_np)
    if coord1.size > 0:
        ymin = float(min(coord1)) / mask_np.shape[0]
        ymax = float(max(coord1) + 1) / mask_np.shape[0]
        xmin = float(min(coord2)) / mask_np.shape[1]
        xmax = float((max(coord2) + 1)) / mask_np.shape[1]

        return np.array([ymin, xmin, ymax, xmax])
    else:
        return np.array([0.0, 0.0, 0.0, 0.0])


def decode_gt_raw_data_into_masks_and_boxes(masks: Dict, img_sizes: Dict):
    """Decods binary segmentation masks into np.arrays and boxes.
    Returns:
        a np.ndarray of the size NxWxH, where W and H is determined from the encoded
        masks; for the None values, zero arrays of size WxH are created. If input
        contains only None values, W=1, H=1.
    """
    segment_masks = []
    segment_boxes = []
    # get the first valid size
    default_size = None
    for segment, img_size in zip(masks, img_sizes):
        if segment != "nan":
            default_size = [
                int(img_size["img_height"]),
                int(img_size["img_width"]),
            ]
            break
    for segment, img_size in zip(masks, img_sizes):
        if segment == "nan":
            # It does not matter which size we pick since no masks will ever be
            # evaluated.
            segment_masks.append(
                np.zeros([1, default_size[0], default_size[1]], dtype=np.uint8)
            )
            segment_boxes.append(np.expand_dims(np.array([0.0, 0.0, 0.0, 0.0]), 0))
        else:
            img_height = int(img_size["img_height"])
            img_width = int(img_size["img_width"])
            compressed_mask = base64.b64decode(segment)
            rle_encoded_mask = zlib.decompress(compressed_mask)
            decoding_dict = {
                "size": [img_height, img_width],
                "counts": rle_encoded_mask,
            }
            mask_tensor = coco_mask.decode(decoding_dict)
            segment_masks.append(np.expand_dims(mask_tensor, 0))
            segment_boxes.append(np.expand_dims(to_normalized_box(mask_tensor), 0))
    return np.concatenate(segment_masks, axis=0), np.concatenate(segment_boxes, axis=0)


def encode_masks(masks: Tensor):
    encoded_mask = []

    for mask in masks:
        mask = np.array(mask).astype(np.uint8)
        mask = np.asfortranarray(mask)
        encoded_mask.append(coco_mask.encode(mask))

    return encoded_mask


def decode_masks(mask: Dict):
    mask_tensor = coco_mask.decode(mask)
    return np.expand_dims(mask_tensor, 0), np.expand_dims(
        to_normalized_box(mask_tensor), 0
    )
