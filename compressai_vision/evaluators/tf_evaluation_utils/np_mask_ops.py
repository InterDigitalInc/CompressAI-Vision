# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
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


"""Operations for [N, height, width] numpy arrays representing masks.

Example mask operations that are supported:
  * Areas: compute mask areas
  * IOU: pairwise intersection-over-union scores
"""

from __future__ import division, print_function

import numpy as np

EPSILON = 1e-7


def area(masks):
    """Computes area of masks.

    Args:
      masks: Numpy array with shape [N, height, width] holding N masks. Masks
        values are of type np.uint8 and values are in {0,1}.

    Returns:
      a numpy array with shape [N*1] representing mask areas.

    Raises:
      ValueError: If masks.dtype is not np.uint8
    """
    if masks.dtype != np.uint8:
        raise ValueError("Masks type should be np.uint8")
    return np.sum(masks, axis=(1, 2), dtype=np.float32)


def intersection(masks1, masks2):
    """Compute pairwise intersection areas between masks.

    Args:
      masks1: a numpy array with shape [N, height, width] holding N masks. Masks
        values are of type np.uint8 and values are in {0,1}.
      masks2: a numpy array with shape [M, height, width] holding M masks. Masks
        values are of type np.uint8 and values are in {0,1}.

    Returns:
      a numpy array with shape [N*M] representing pairwise intersection area.

    Raises:
      ValueError: If masks1 and masks2 are not of type np.uint8.
    """
    if masks1.dtype != np.uint8 or masks2.dtype != np.uint8:
        raise ValueError("masks1 and masks2 should be of type np.uint8")
    n = masks1.shape[0]
    m = masks2.shape[0]
    answer = np.zeros([n, m], dtype=np.float32)
    for i in np.arange(n):
        for j in np.arange(m):
            answer[i, j] = np.sum(np.minimum(masks1[i], masks2[j]), dtype=np.float32)
    return answer


def iou(masks1, masks2):
    """Computes pairwise intersection-over-union between mask collections.

    Args:
      masks1: a numpy array with shape [N, height, width] holding N masks. Masks
        values are of type np.uint8 and values are in {0,1}.
      masks2: a numpy array with shape [M, height, width] holding N masks. Masks
        values are of type np.uint8 and values are in {0,1}.

    Returns:
      a numpy array with shape [N, M] representing pairwise iou scores.

    Raises:
      ValueError: If masks1 and masks2 are not of type np.uint8.
    """
    if masks1.dtype != np.uint8 or masks2.dtype != np.uint8:
        raise ValueError("masks1 and masks2 should be of type np.uint8")
    intersect = intersection(masks1, masks2)
    area1 = area(masks1)
    area2 = area(masks2)
    union = np.expand_dims(area1, axis=1) + np.expand_dims(area2, axis=0) - intersect
    return intersect / np.maximum(union, EPSILON)


def ioa(masks1, masks2):
    """Computes pairwise intersection-over-area between box collections.

    Intersection-over-area (ioa) between two masks, mask1 and mask2 is defined as
    their intersection area over mask2's area. Note that ioa is not symmetric,
    that is, IOA(mask1, mask2) != IOA(mask2, mask1).

    Args:
      masks1: a numpy array with shape [N, height, width] holding N masks. Masks
        values are of type np.uint8 and values are in {0,1}.
      masks2: a numpy array with shape [M, height, width] holding N masks. Masks
        values are of type np.uint8 and values are in {0,1}.

    Returns:
      a numpy array with shape [N, M] representing pairwise ioa scores.

    Raises:
      ValueError: If masks1 and masks2 are not of type np.uint8.
    """
    if masks1.dtype != np.uint8 or masks2.dtype != np.uint8:
        raise ValueError("masks1 and masks2 should be of type np.uint8")
    intersect = intersection(masks1, masks2)
    areas = np.expand_dims(area(masks2), axis=0)
    return intersect / (areas + EPSILON)
