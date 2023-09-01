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

import logging
import math
from pathlib import Path
from typing import Dict, Tuple

import torch.nn as nn

from compressai_vision.registry import register_codec


@register_codec("bypass")
class Bypass(nn.Module):
    """Does no encoding/decoding whatsoever. Use for debugging."""

    def __init__(self, **kwargs):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.qp = None
        self.eval_encode = kwargs["eval_encode"]
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
    ) -> Dict:
        """
        Bypass encoder
        Returns the input and calculates its raw size
        """
        del file_prefix  # used in other codecs that write bitstream files
        del bitstream_name  # used in other codecs that write bitstream files
        del codec_output_dir  # used in other codecs that write log files

        total_elements = 0
        for _, ft in input["data"].items():
            N = ft.size(0)
            total_elements += _number_of_elements(ft.size())

        # write input
        total_bytes = total_elements * 4  # 32-bit floating

        total_bytes = [total_bytes / N] * H

        return {
            "bytes": total_bytes,
            "bitstream": input,
        }

    def decode(
        self,
        input: Dict,
        codec_output_dir: str = "",
        file_prefix: str = "",
    ):
        del file_prefix  # used in other codecs that write log files
        del codec_output_dir  # used in other codecs that write log files
        return input


def _number_of_elements(data: Tuple):
    return math.prod(data)
