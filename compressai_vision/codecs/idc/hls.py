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

from typing import Any, Dict, List, Tuple, cast

from compressai_vision.codecs.encdec_utils import *

from .common import FeatureTensorCodingType

__all__ = ["SequenceParameterSet", "parse_feature_tensor_coding_type"]


class SequenceParameterSet:
    def __init__(self):
        self.org_input_height = self.org_input_width = -1
        self.input_height = self.input_width = -1
        self.size_of_feature_set = -1

        # sanity check
        self.shapes_of_features = []
        self.group_order_flag = False
        self.sps_id = -1

    def digest(self, **kwargs):
        assert set(["input_size", "org_input_size", "data"]).issubset(kwargs)

        shapes_of_features = []
        for item in kwargs["data"].values():
            assert item.dim() == 4
            shapes_of_features.append(item.shape[1:])

        self.org_input_height, self.org_input_width = kwargs["org_input_size"].values()
        self.input_height, self.input_width = kwargs["input_size"][0]
        self.size_of_feature_set = len(shapes_of_features)

        # sanity check
        self.shapes_of_features = []
        for shape in shapes_of_features:
            C, H, W = shape

            assert (C % 16) == 0 and C >= 16 and C <= 4080
            assert H <= 2040
            assert W <= 2040

            # any restriction on spatial resolution for tensor?

            dshape = {
                "num_channels": C,  # 8-bit num_channels_shift_right_4
                "channel_height": H,  # 8-bit channel_height_shift_right_3
                "channel_width": W,  # 8-bit channel_width_shift_right_3
            }
            self.shapes_of_features.append(dshape)

        # Suppose that the group order information can be inferred
        # in case that NN-part 1 and NN-part 2 are pre-configured regarding the split channels order.
        self.group_order_flag = False

        self.sps_id = -1

    def assign_id(self, sps_id):
        self.sps_id = sps_id

    def register_group_order(self, group_orders: Dict):
        self.group_order_flag = True

        raise NotImplementedError

    def write(
        self,
        fd,
        nbframes,
        qp,
        qp_density,
        is_downsampled,
        dc_qp_offset,
        dc_qp_density_offset,
    ):
        byte_cnt = 0
        # encode header (sequence level)

        # please review [downsample flag] TODO: @eimran mask with other flag (if possible)
        byte_cnt += write_uchars(fd, (BoolConvert(is_downsampled),))

        # write original input resolution
        byte_cnt += write_uints(fd, (self.org_input_height, self.org_input_width))

        # write input resolution
        byte_cnt += write_uints(fd, (self.input_height, self.input_width))

        # write number of feature sets (1 byte)
        byte_cnt += write_uchars(fd, (self.size_of_feature_set,))

        # write feature tensor dims (4 bytes each - temp): nb_channels, H, W
        for dshape in self.shapes_of_features:
            byte_cnt += write_uints(fd, tuple(dshape.values()))

        # temporary syntax
        # decoder needs number of frames (temp)
        self.nbframes = nbframes
        byte_cnt += write_uints(fd, (nbframes,))

        # encode QP in bitstream (This should be part of picture and layer parameter set )
        self.qp = qp
        byte_cnt += write_uchars(fd, ((qp + 128),))

        self.qp_density = qp_density
        byte_cnt += write_uchars(fd, (qp_density,))

        # DC qp and density
        self.dc_qp_offset = dc_qp_offset
        byte_cnt += write_uchars(fd, (dc_qp_offset + 128,))  # QP

        self.dc_qp_density_offset = dc_qp_density_offset
        byte_cnt += write_uchars(fd, (dc_qp_density_offset,))  # QP_DENSITY

        return byte_cnt

    def read(self, fd):
        byte_cnt = 0
        # encode header (sequence level)

        # read donwsample flag
        self.is_downsampled = read_uchars(fd, 1)[0]

        # read original input resolution
        self.org_input_height, self.org_input_width = read_uints(fd, 2)

        # read input resolution
        self.input_height, self.input_width = read_uints(fd, 2)

        # write number of feature sets (1 byte)
        self.size_of_feature_set = read_uchars(fd, 1)[0]

        # read feature tensor dims (4 bytes each - temp): nb_channels, H, W
        for _ in range(self.size_of_feature_set):
            C, H, W = read_uints(fd, 3)

            assert (C % 16) == 0 and C >= 16 and C <= 4080
            assert H <= 2040
            assert W <= 2040

            # any restriction on spatial resolution for tensor?
            dshape = {
                "num_channels": C,  # 8-bit num_channels_shift_right_4
                "channel_height": H,  # 8-bit channel_height_shift_right_3
                "channel_width": W,  # 8-bit channel_width_shift_right_3
            }
            self.shapes_of_features.append(dshape)

        # temporary syntax
        # decoder needs number of frames (temp)
        nbframes = read_uints(fd, 1)[0]
        self.nbframes = nbframes

        # encode QP in bitstream (This should be part of picture and layer parameter set )
        qp = read_uchars(fd, 1)[0]
        self.qp = int(qp - 128)

        qp_density = read_uchars(fd, 1)[0]
        self.qp_density = qp_density

        # for intra DC
        dc_qp_offset = read_uchars(fd, 1)[0]
        self.dc_qp_offset = int(dc_qp_offset - 128)

        dc_qp_density_offset = read_uchars(fd, 1)[0]
        self.dc_qp_density_offset = dc_qp_density_offset


def parse_feature_tensor_coding_type(fd):
    val = read_uchars(fd, 1)[0]
    return FeatureTensorCodingType(val)
