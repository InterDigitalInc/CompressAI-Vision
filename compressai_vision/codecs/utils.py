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
