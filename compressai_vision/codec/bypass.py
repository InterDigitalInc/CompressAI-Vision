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

import io
import logging
import math
import os
from pathlib import Path
from typing import Dict, Tuple

import torch

from compressai_vision.registry import register_codec

from .base import EncoderDecoder
from .syntax.readwrite import write_uints


@register_codec("bypass")
class VoidEncoderDecoder(EncoderDecoder):
    """Does no encoding/decoding whatsoever.  Use for debugging."""

    def __init__(self, **kwargs):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.reset()

        self.output_dir = self._create_folder(kwargs["output_dir"])

    def reset(self):
        """Reset the internal state of the encoder & decoder, if any"""
        self.cc = 0

    def encode(self, input: Dict, tag: str = None):
        """
        :param input: input data in dictonary
            By default, the input data must include following keywords
            'data' : A dictionary variable conveys input data in tensor to compress associated with keyword
            'input_size' : The size of the input to the pre-inference module, which might be properly resized to be fed into the model
            'org_input_size' : The size of the original input data as is

        :param tag: a string that can be used to identify & cache bitstream

        Returns

        Compress the input, write a bitstream

        Returns a list of bits per frame and a path for the bitstream
        """
        # Not really write the tensors into a file
        # file_path = os.path.join(self.output_dir, f"{tag}.bin")

        # with Path(file_path).open("wb") as fd:
        #    # write minimum header
        #    # write original image size
        #    write_uints(fd, self.getOrgInputSize(input["org_input_size"]))
        #    # write input size
        #    write_uints(fd, self.getInputSize(input["input_size"]))

        total_elements = 0
        for ft in input["data"].values():
            total_elements += self._number_of_elements(ft.size())

        # write input
        total_bytes = total_elements * 4  # supposed 32-bit floating

        # For the bypass codec, the bitstream field points to input dictionary itself
        return {
            "bytes": [
                total_bytes,
            ],
            "bitstream": input,
        }

    def decode(self, input, tag: str = None):
        return input

    @staticmethod
    def _number_of_elements(data: Tuple):
        return math.prod(data)
