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

import numpy as np

from compressai_vision.codecs.encdec_utils import *

from .common import FeatureTensorCodingType
from .entropy import decode_integers_deepcabac, encode_integers_deepcabac

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
        downscale_flag,
        qp,
        qp_density,
        layer_qp_offsets,
        dc_qp_offset,
        dc_qp_density_offset,
    ):
        byte_cnt = 0
        # encode header (sequence level)

        # please review [downsample flag] TODO: @eimran mask with other flag (if possible)
        tools_flag = 0
        self.downscale_flag = int(downscale_flag)
        tools_flag |= self.downscale_flag << 6

        byte_cnt += write_uchars(fd, (tools_flag,))

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

        self.layer_qp_offsets = layer_qp_offsets
        for layer_qp_offset in layer_qp_offsets:
            byte_cnt += write_uchars(fd, (layer_qp_offset + 128,))

        # DC qp and density
        self.dc_qp_offset = dc_qp_offset
        byte_cnt += write_uchars(fd, (dc_qp_offset + 128,))  # QP

        self.dc_qp_density_offset = dc_qp_density_offset
        byte_cnt += write_uchars(fd, (dc_qp_density_offset,))  # QP_DENSITY

        return byte_cnt

    def read(self, fd):
        byte_cnt = 0
        # encode header (sequence level)

        # read tools flag
        tools_flag = read_uchars(fd, 1)[0]
        self.downscale_flag = BoolConvert(((tools_flag >> 6) & 0x01))

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

        # qp per layer
        self.layer_qp_offsets = []
        for _ in range(self.size_of_feature_set - 1):
            layer_qp_offset = read_uchars(fd, 1)[0]
            self.layer_qp_offsets.append(int(layer_qp_offset - 128))

        # for intra DC
        dc_qp_offset = read_uchars(fd, 1)[0]
        self.dc_qp_offset = int(dc_qp_offset - 128)

        dc_qp_density_offset = read_uchars(fd, 1)[0]
        self.dc_qp_density_offset = dc_qp_density_offset


class FeatureTensorsHeader:
    def __init__(
        self,
        sps: SequenceParameterSet,
        ctype: FeatureTensorCodingType = None,
        tags: List = None,
    ):
        self.sps = sps
        self._coding_type = ctype
        self._ftoc = -1
        self._coding_modes = None
        self._scales_info = None
        self._tags = tags

    @property
    def ftoc(self):
        return self._ftoc

    def set_ftoc(self, num: int):
        self._ftoc = num

    def coding_modes(self, tag):
        return self._coding_modes[tag]

    def get_coding_modes(self):
        return self._coding_modes

    def set_coding_modes(self, coding_modes: Dict):
        self._coding_modes = coding_modes

    @property
    def coding_type(self):
        return self._coding_type

    def set_coding_type(self, val: int):
        self._coding_type = FeatureTensorCodingType(val)

    def set_scales_info(self, scales_info: Dict):
        self._scales_info = scales_info

    def get_scales_info(self):
        return self._scales_info

    def scale(self, tag):
        return self._scales_info[tag]

    def get_split_layer_tags(self) -> List:
        return self._tags

    def write(self, fd):
        byte_cnt = write_uchars(fd, (self.coding_type.value,))

        if self.coding_type == FeatureTensorCodingType.I_TYPE:
            for tag in self.get_split_layer_tags():
                coding_modes = self.coding_modes(tag)
                scale_minus_1 = self.scale(tag) - 1

                # encoding channels_coding_modes with cabac
                indexes = np.array(coding_modes + 1, dtype=np.int32)
                byte_cnt += encode_integers_deepcabac(indexes, fd)

                # TODO bit wise sps + byte alignment
                if self.sps.downscale_flag:
                    byte_cnt += write_uchars(
                        fd,
                        [
                            scale_minus_1,
                        ],
                    )

        return byte_cnt

    def read(self, fd):
        val = read_uchars(fd, 1)[0]
        self.set_coding_type(val)

        if self.coding_type == FeatureTensorCodingType.I_TYPE:
            coding_modes_d = {}
            scale_info_d = {}
            for e, shape in enumerate(self.sps.shapes_of_features):
                C = shape["num_channels"]
                coding_modes_d[e] = decode_integers_deepcabac(fd, C)

                scale = 1
                if self.sps.downscale_flag:
                    # it would be better to read by bit
                    scale_minus_1 = read_uchars(fd, 1)[0]
                    scale = scale_minus_1 + 1

                scale_info_d[e] = scale

            self.set_coding_modes(coding_modes_d)
            self.set_scales_info(scale_info_d)

        return 0
