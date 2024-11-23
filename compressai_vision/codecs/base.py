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
import math
from typing import Dict, Tuple

import torch.nn as nn

from compressai_vision.registry import register_codec
from compressai_vision.utils import time_measure


@register_codec("bypass")
class Bypass(nn.Module):
    """Does no encoding/decoding whatsoever. Use for debugging."""

    def __init__(self, **kwargs):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.qp = None
        self.eval_encode = kwargs["eval_encode"]
        self.nbit_quant = kwargs["encoder_config"]["nbit_quant"]
        # output_dir = Path(kwargs["output_dir"])
        # if not output_dir.is_dir():
        #     self.logger.info(f"creating output folder: {output_dir}")
        #     output_dir.mkdir(parents=True, exist_ok=True)

    @property
    def qp_value(self):
        return self.qp

    @property
    def eval_encode_type(self):
        return self.eval_encode

    def encode(
        self,
        input: Dict,
        codec_output_dir: str = "",
        bitstream_name: str = "",
        file_prefix: str = "",
        remote_inference=False,
    ) -> Dict:
        """
        Bypass encoder
        Returns the input and calculates its raw size
        """
        del file_prefix  # used in other codecs that write bitstream files
        del bitstream_name  # used in other codecs that write bitstream files
        del codec_output_dir  # used in other codecs that write log files

        mac_calculations = None  # no NN-related complexity calculation

        if remote_inference is True:
            org_fH = input["org_input_size"]["height"]
            org_fW = input["org_input_size"]["width"]

            num_elements = org_fH * org_fW
            num_frames = len(input["file_names"])

            enc_time = 0

            return (
                {
                    "bytes": [num_elements] * num_frames,
                    "bitstream": input,
                },
                enc_time,
                mac_calculations,
            )

        # for n-bit quantization error experiments
        max_lvl = ((2**self.nbit_quant) - 1) if self.nbit_quant != -1 else None

        total_elements = 0
        start_time = time_measure()
        for tag, ft in input["data"].items():
            N = ft.size(0)
            total_elements += _number_of_elements(ft.size())

            # for n-bit quantization error experiments
            if max_lvl is not None:
                minv = ft.min()
                maxv = ft.max()

                quant_ft = (ft - minv) / (maxv - minv)
                quant_ft = quant_ft.clamp_(0, 1) * max_lvl
                quant_ft = quant_ft.round() / max_lvl
                quant_ft = (quant_ft * (maxv - minv)) + minv

                input["data"][tag] = quant_ft

        # write input
        total_bytes = total_elements * 4  # 32-bit floating

        total_bytes = [total_bytes / N] * N

        enc_time = {
            "bypass": time_measure() - start_time,
        }

        return (
            {
                "bytes": total_bytes,
                "bitstream": input,
            },
            enc_time,
            mac_calculations,
        )

    def decode(
        self,
        input: Dict,
        codec_output_dir: str = "",
        file_prefix: str = "",
        org_img_size: Dict = None,
        remote_inference=False,
        vcm_mode=False,
    ):
        del org_img_size
        del file_prefix  # used in other codecs that write log files
        del codec_output_dir  # used in other codecs that write log files

        dec_time = {"bypass": 0}
        mac_calculations = None  # no NN-related complexity calculation

        if remote_inference:
            assert "file_names" in input

        return input, dec_time, mac_calculations


def _number_of_elements(data: Tuple):
    return math.prod(data)
