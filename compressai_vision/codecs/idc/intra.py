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

from typing import Any, Dict, List

import deepCABAC
import numpy as np
import torch

from compressai_vision.codecs.encdec_utils import *

from .entropy import dequantize, dequantize_and_decode, quantize_and_encode
from .hls import FeatureTensorsHeader, SequenceParameterSet
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


def intra_coding(
    sps: SequenceParameterSet,
    ftHeader: FeatureTensorsHeader,
    ftensors: Dict,
    decoded_buffer: List,
    bitstream_fd: Any,
):
    # deepCABAC decoder instance for reconstruction
    decoder = deepCABAC.Decoder()

    byte_cnt = 0

    recon_ftensors = {}
    layer_idx = 0
    for tag, ftensor in ftensors.items():
        # center data and get DC values
        dc, centered_ftensor = _center_data(ftensor)

        (
            nb_bytes_dc,
            byte_array,
            quantized_dc,
        ) = quantize_and_encode(
            dc,
            sps.qp + sps.dc_qp_offset,
            sps.qp_density + sps.dc_qp_density_offset,
        )

        # reconstruction dc
        decoder.initCtxModels(10)
        dequantized_dc = dequantize(
            decoder,
            quantized_dc,
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
        ) = quantize_and_encode(centered_ftensor, sps.qp, sps.qp_density)

        byte_cnt += write_uints(bitstream_fd, (nb_bytes_tensor,))
        byte_cnt += nb_bytes_tensor
        bitstream_fd.write(byte_array)

        # reconstruct features
        decoder.initCtxModels(10)
        dequantized_ftensor = dequantize(
            decoder, quantized_ftensor, sps.qp, sps.qp_density
        )

        # add dc values
        decoded_ftensor = _add_dc_values(dequantized_ftensor, dequantized_dc)
        recon_ftensors[tag] = decoded_ftensor

        layer_idx += 1

    return byte_cnt, recon_ftensors


def intra_decoding(
    sps: SequenceParameterSet,
    ftHeader: FeatureTensorsHeader,
    decoded_buffer: List,
    bitstream_fd: Any,
):
    recon_ftensors = {}
    recon_ftensors_suppressed = {}
    # all_scales_for_layers = {}
    for e, shape_of_ftensor in enumerate(sps.shapes_of_features):
        C, H, W = shape_of_ftensor.values()

        channel_coding_modes = ftHeader.coding_modes(e)
        scale = ftHeader.scale(e)

        self_coded_labels = np.where(channel_coding_modes == -1)[0]
        nb_groups = max(channel_coding_modes) + 1

        nb_channels_coded_ftensor = len(self_coded_labels) + nb_groups

        # decoding DCs
        byte_to_read = read_uints(bitstream_fd, 1)[0]
        byte_array = bitstream_fd.read(byte_to_read)
        dequantized_dc = dequantize_and_decode(
            (nb_channels_coded_ftensor),
            byte_array,
            sps.qp + sps.dc_qp_offset,
            sps.qp_density + sps.dc_qp_density_offset,
        )

        # decode centered channels
        byte_to_read = read_uints(bitstream_fd, 1)[0]
        byte_array = bitstream_fd.read(byte_to_read)

        qp_layer = sps.qp
        if e > 0:
            qp_layer += sps.layer_qp_offsets[e - 1]

        decoded_ftensor = dequantize_and_decode(
            (nb_channels_coded_ftensor, H // scale, W // scale),
            byte_array,
            qp_layer,
            sps.qp_density,
        )

        # add dc values
        decoded_ftensor = _add_dc_values(decoded_ftensor, dequantized_dc)

        recon_ftensors_suppressed[e] = decoded_ftensor

        recon_ftensors[e] = rebuild_ftensor(
            channel_coding_modes,
            decoded_ftensor,
            scale,
            (C, H, W),
        )

    return recon_ftensors, recon_ftensors_suppressed
