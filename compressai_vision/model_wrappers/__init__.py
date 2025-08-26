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

from .detectron2 import (
    BaseWrapper,
    faster_rcnn_R_50_FPN_3x,
    faster_rcnn_X_101_32x8d_FPN_3x,
    mask_rcnn_R_50_FPN_3x,
    mask_rcnn_X_101_32x8d_FPN_3x,
    panoptic_rcnn_R_101_FPN_3x,
)
from .jde import jde_1088x608
from .rtmo import rtmo_multi_person_pose_estimation
from .sam import (
    sam_vit_b_01ec64,
    sam_vit_h_4b8939,
    sam_vit_l_0b3195,
)
from .yolox import yolox_darknet53

__all__ = [
    "BaseWrapper",
    "faster_rcnn_X_101_32x8d_FPN_3x",
    "mask_rcnn_X_101_32x8d_FPN_3x",
    "faster_rcnn_R_50_FPN_3x",
    "mask_rcnn_R_50_FPN_3x",
    "panoptic_rcnn_R_101_FPN_3x",
    "jde_1088x608",
    "yolox_darknet53",
    "rtmo_multi_person_pose_estimation",
    "sam_vit_h_4b8939",
    "sam_vit_b_01ec64",
    "sam_vit_l_0b3195",
]
