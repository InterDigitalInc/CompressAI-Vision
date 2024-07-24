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

import logging
from typing import Any, Dict, List

import torch.nn as nn
from torch import Tensor


class BaseWrapper(nn.Module):
    """NOTE: virtual class to build *your* wrapper and interface with compressai_vision

    An instance of this class helps you to wrap an off-the-shelf model so that the wrapped model can behave in various modes such as "full" and "partial" to process the input frames.
    """

    def __init__(self) -> None:
        super().__init__()
        self.logger = logging.getLogger(self.__class__.__name__)

    def input_to_features(self, x, device: str) -> Dict:
        """Computes deep features at the intermediate layer(s) all the way from the input"""
        raise NotImplementedError

    def features_to_output(self, x: Dict, device: str):
        """Complete the downstream task from the intermediate deep features"""
        raise NotImplementedError

    def deeper_features_for_accuracy_proxy(self, x: Dict):
        """
        compute accuracy proxy at the deeper layer than NN-Part1
        """
        raise NotImplementedError

    def forward(self, x, input_map_function):
        """Complete the downstream task with end-to-end manner all the way from the input"""
        raise NotImplementedError

    @property
    def cfg(self):
        return None

    @property
    def pretrained_weight_path(self):
        return self.model_info["weights"]

    @property
    def model_cfg_path(self):
        return self.model_info["cfg"]
