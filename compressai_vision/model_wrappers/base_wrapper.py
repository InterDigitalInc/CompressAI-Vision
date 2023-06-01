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

from typing import Dict, List

import torch.nn as nn


class BaseWrapper(nn.Module):
    """NOTE: virtual class to build *your* wrapper and interface with compressai_vision

    An instance of this class helps you to wrap an off-the-shelf model so that the wrapped model can behave in various modes such as "full" and "partial" to process the input frames.
    """

    def input_to_features(self, x):
        """Computes deep features at the intermediate layer(s) all the way from the input"""
        raise NotImplementedError

    def features_to_output(self, x):
        """Complete the downstream task from the intermediate deep features"""
        raise NotImplementedError
    def input_to_feature_pyramid(self, x):
        """Computes and return feture pyramid ['p2', 'p3', 'p4', 'p5'] all the way from the input"""
        raise NotImplementedError

    def feature_pyramid_to_output(self, x, org_img_size: Dict, input_img_size: List):
        """Complete the downstream task from the feature pyramid ['p2', 'p3', 'p4', 'p5']"""
        raise NotImplementedError

    def forward(self, x):
        """Complete the downstream task with end-to-end manner all the way from the input"""
        raise NotImplementedError

    def reshape_feature_to_frame(self, x):
        """rehape feature tensor channels to a frame"""
        raise NotImplementedError

    def reshape_frame_to_feature(self, x, tensor_shape):
        """reshape a frame of channels into feature tensor(s)"""
        raise NotImplementedError

    def reshape_feature_pyramid_to_frame(self, x):
        """rehape the feature pyramid to a frame"""
        raise NotImplementedError

    def reshape_frame_to_feature_pyramid(self, x, tensor_shape):
        """reshape a frame of channels into the feature pyramid"""
        raise NotImplementedError
