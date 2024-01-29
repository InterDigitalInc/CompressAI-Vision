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

"""Numpy BoxMaskList classes and functions."""

from __future__ import division, print_function

import numpy as np

from ..tf_evaluation_utils import np_box_list


class BoxMaskList(np_box_list.BoxList):
    """Convenience wrapper for BoxList with masks.

    BoxMaskList extends the np_box_list.BoxList to contain masks as well.
    In particular, its constructor receives both boxes and masks. Note that the
    masks correspond to the full image.
    """

    def __init__(self, box_data, mask_data):
        """Constructs box collection.

        Args:
          box_data: a numpy array of shape [N, 4] representing box coordinates
          mask_data: a numpy array of shape [N, height, width] representing masks
            with values are in {0,1}. The masks correspond to the full
            image. The height and the width will be equal to image height and width.

        Raises:
          ValueError: if bbox data is not a numpy array
          ValueError: if invalid dimensions for bbox data
          ValueError: if mask data is not a numpy array
          ValueError: if invalid dimension for mask data
        """
        super(BoxMaskList, self).__init__(box_data)
        if not isinstance(mask_data, np.ndarray):
            raise ValueError("Mask data must be a numpy array.")
        if len(mask_data.shape) != 3:
            raise ValueError("Invalid dimensions for mask data.")
        if mask_data.dtype != np.uint8:
            raise ValueError("Invalid data type for mask data: uint8 is required.")
        if mask_data.shape[0] != box_data.shape[0]:
            raise ValueError("There should be the same number of boxes and masks.")
        self.data["masks"] = mask_data

    def get_masks(self):
        """Convenience function for accessing masks.

        Returns:
          a numpy array of shape [N, height, width] representing masks
        """
        return self.get_field("masks")
