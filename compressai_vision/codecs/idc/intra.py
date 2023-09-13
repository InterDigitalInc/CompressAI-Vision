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

from typing import Any, Dict, Tuple

import deepCABAC
import numpy as np
import torch
from scipy.stats import norm

from compressai_vision.codecs.encdec_utils import *

from .common import FeatureTensorCodingType
from .hls import SequenceParameterSet
from .tools import rebuild_ftensor

__all__ = [
    "intra_coding",
    "intra_decoding",
]


def _center_data(clipped_ftensor):
    dc = torch.mean(clipped_ftensor, dim=(1, 2))
    center_data = torch.sub(clipped_ftensor, dc[:, None, None])
    return dc, center_data


def _add_dc_values(ftensor, dc):
    return torch.add(ftensor, dc[:, None, None])


def _quantize_and_encode(channels, qp, qp_density, maxValue=-1):
    encoder = deepCABAC.Encoder()  # deepCABAC Encoder
    encoder.initCtxModels(10, 0)  # TODO: check nb ctx

    quantizedValues = np.zeros(channels.shape, dtype=np.int32)
    encoder.quantFeatures(
        channels.detach().cpu().numpy(), quantizedValues, qp_density, qp, 0
    )  # TODO: @eimran change scan order and qp method

    encoder.encodeFeatures(quantizedValues, 0, maxValue)
    bs = bytearray(encoder.finish().tobytes())
    total_bytes_spent = len(bs)
    # return quantized values for debugging
    return total_bytes_spent, bs, quantizedValues


def _dequantize_and_decode(data_shape: Tuple, bs, qp, qp_density):
    # tC, tH, tW = data_shape

    assert isinstance(bs, (bytearray, bytes))

    if not isinstance(bs, bytearray):
        bs = bytearray(bs)

    # need to decode max_value
    max_value = -1

    decoder = deepCABAC.Decoder()
    decoder.initCtxModels(10)
    decoder.setStream(bs)
    quantizedValues = np.zeros(data_shape, dtype=np.int32)
    decoder.decodeFeatures(quantizedValues, 0, max_value)
    # print(quantizedValues[0, :10, :10])
    recon_features = np.zeros(quantizedValues.shape, dtype=np.float32)
    decoder.dequantFeatures(
        recon_features, quantizedValues, qp_density, qp, 0
    )  # TODO: @eimran send qp with bitstream
    # print(recon_features[0, :10, :10])

    return recon_features


def encode_integers_deepcabac(array: np.array, bitstream_fd):
    encoder = deepCABAC.Encoder()
    encoder.initCtxModels(10, 1)  # TODO check args
    encoder.encodeFeatures(array, 0, 0)
    bs = bytearray(encoder.finish().tobytes())
    byte_cnt = write_uints(bitstream_fd, (len(bs),))
    bitstream_fd.write(bs)
    return byte_cnt + len(bs)


def intra_coding(
    sps: SequenceParameterSet,
    ftensors: Dict,
    all_coding_groups: Dict,
    scales_for_layers: Dict,
    bitstream_fd: Any,
):
    byte_cnt = 0

    byte_cnt += write_uchars(bitstream_fd, (FeatureTensorCodingType.I_TYPE.value,))

    recon_ftensors = {}
    layer_idx = 0
    for tag, ftensor in ftensors.items():
        coding_groups = all_coding_groups[tag]
        scale_minus_1 = scales_for_layers[tag] - 1

        # encoding channels_coding_modes with cabac
        indexes = np.array(coding_groups + 1, dtype=np.int32)
        byte_cnt += encode_integers_deepcabac(indexes, bitstream_fd)

        # TODO bit wise sps + byte alignment
        if sps.downscale_flag:
            byte_cnt += write_uchars(
                bitstream_fd,
                [
                    scale_minus_1,
                ],
            )

        # center data and get DC values
        dc, centered_ftensor = _center_data(ftensor)

        (
            nb_bytes_dc,
            byte_array,
            quantized_dc,
        ) = _quantize_and_encode(
            dc,
            sps.qp + sps.dc_qp_offset,
            sps.qp_density + sps.dc_qp_density_offset,
        )

        byte_cnt += write_uints(bitstream_fd, (nb_bytes_dc,))
        byte_cnt += nb_bytes_dc
        bitstream_fd.write(byte_array)
        # end

        layer_qp = sps.qp
        if layer_idx > 0:
            layer_qp += sps.layer_qp_offsets[layer_idx - 1]
        (
            nb_bytes_tensor,
            byte_array,
            quantized_ftensor,
        ) = _quantize_and_encode(centered_ftensor, sps.qp, sps.qp_density)

        byte_cnt += write_uints(bitstream_fd, (nb_bytes_tensor,))
        byte_cnt += nb_bytes_tensor
        bitstream_fd.write(byte_array)

        # reconstruct features
        decoder = deepCABAC.Decoder()
        decoder.initCtxModels(10)

        dequantized_ftensor = np.zeros(quantized_ftensor.shape, dtype=np.float32)
        decoder.dequantFeatures(
            dequantized_ftensor,
            quantized_ftensor,
            sps.qp_density,
            sps.qp,
            0,
        )

        recon_ftensors[tag] = dequantized_ftensor

        layer_idx += 1

    return byte_cnt, recon_ftensors


def decode_integers_deepcabac(bitstream_fd, array_size) -> np.array:
    bytes_to_read = read_uints(bitstream_fd, 1)[0]
    byte_array = bytearray(bitstream_fd.read(bytes_to_read))
    decoder = deepCABAC.Decoder()
    decoder.initCtxModels(10)
    decoder.setStream(byte_array)
    decoded_values = np.zeros(array_size, dtype=np.int32)
    decoder.decodeFeatures(decoded_values, 0, 0)
    # shift modes so -1 = self-coded
    decoded_values -= 1
    return decoded_values


def intra_decoding(sps: SequenceParameterSet, bitstream_fd: Any):
    recon_ftensors = {}
    # all_scales_for_layers = {}
    for e, shape_of_ftensor in enumerate(sps.shapes_of_features):
        C, H, W = shape_of_ftensor.values()

        # decode channel coding modes
        channel_coding_modes = decode_integers_deepcabac(bitstream_fd, C)

        scale = 1
        if sps.downscale_flag:
            # it would be better to read by bit
            scale_minus_1 = read_uchars(bitstream_fd, 1)[0]
            scale = scale_minus_1 + 1

        self_coded_labels = np.where(channel_coding_modes == -1)[0]
        nb_groups = max(channel_coding_modes) + 1

        nb_channels_coded_ftensor = len(self_coded_labels) + nb_groups

        # decoding DCs
        byte_to_read = read_uints(bitstream_fd, 1)[0]
        byte_array = bitstream_fd.read(byte_to_read)
        dequantized_dc = _dequantize_and_decode(
            (nb_channels_coded_ftensor),
            byte_array,
            sps.qp + sps.dc_qp_offset,
            sps.qp_density + sps.dc_qp_density_offset,
        )
        dequantized_dc = torch.from_numpy(dequantized_dc)

        # decode centered channels
        byte_to_read = read_uints(bitstream_fd, 1)[0]
        byte_array = bitstream_fd.read(byte_to_read)

        qp_layer = sps.qp
        if e > 0:
            qp_layer += sps.layer_qp_offsets[e - 1]

        decoded_ftensor = torch.from_numpy(
            _dequantize_and_decode(
                (nb_channels_coded_ftensor, H // scale, W // scale),
                byte_array,
                qp_layer,
                sps.qp_density,
            )
        )

        # add dc values
        decoded_ftensor = _add_dc_values(decoded_ftensor, dequantized_dc)

        recon_ftensors[e] = rebuild_ftensor(
            channel_coding_modes,
            decoded_ftensor,
            scale,
            (C, H, W),
        )

    return recon_ftensors
