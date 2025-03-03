# Copyright (c) 2025, InterDigital Communications, Inc
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


import torch.nn as nn

from .squeeze_base import squeeze_base


# for YOLOX-Darknet53
class three_convs_at_l13(squeeze_base):
    def __init__(self, C0, C1, C2, C3):
        super().__init__(C0, C1, C2, C3)

        self.fw_block = nn.Sequential(
            nn.Conv2d(
                in_channels=C0, out_channels=C1, kernel_size=3, padding=1, stride=1
            ),
            nn.PReLU(),
            nn.Conv2d(
                in_channels=C1, out_channels=C2, kernel_size=3, padding=1, stride=2
            ),
            nn.PReLU(),
            nn.Conv2d(
                in_channels=C2, out_channels=C3, kernel_size=1, padding=0, stride=1
            ),
            nn.SiLU(inplace=True),
        )

        self.bw_block = nn.Sequential(
            nn.Conv2d(
                in_channels=C3, out_channels=C2, kernel_size=3, padding=1, stride=1
            ),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.PReLU(),
            nn.Conv2d(
                in_channels=C2, out_channels=C1, kernel_size=3, padding=1, stride=1
            ),
            nn.PReLU(),
            nn.Conv2d(
                in_channels=C1, out_channels=C0, kernel_size=1, padding=0, stride=1
            ),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
        )

    @property
    def address(self):
        return "https://dspub.blob.core.windows.net/compressai-vision/split_squeezes/yolox_darknet53/three_convs_squeeze_at_l13_of_yolox_darknet53-f78179c1.pth"

    def squeeze_(self, x):
        return self.fw_block(x)

    def expand_(self, x):
        return self.bw_block(x)

    def forward(self, x):
        y = self.fw_block(x)
        est_x = self.bw_block(y)
        return est_x
