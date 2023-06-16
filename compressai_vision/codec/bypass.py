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
from typing import Dict

from compressai_vision.registry import register_codec

from .base import EncoderDecoder


@register_codec("bypass")
class VoidEncoderDecoder(EncoderDecoder):
    """Does no encoding/decoding whatsoever.  Use for debugging."""

    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.reset()

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

        Returns a list of bits per frame along with input frame size and a path for the bitstream
        """

        """
        
        shape = out["shape"]

        with Path(output).open("wb") as f:
            write_uchars(f, codec.codec_header)
            # write original image size
            write_uints(f, (h, w))
            # write original bitdepth
            write_uchars(f, (bitdepth,))
            # write shape and number of encoded latents
            write_body(f, shape, out["strings"])

        size = filesize(output)
        bpp = float(size) * 8 / (h * w)

        """
        # write input

        return 0, input

    def decode(self, bitstream_path):
        return 0
