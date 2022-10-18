# Copyright (c) 2022, InterDigital Communications, Inc
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

import logging, math
import torch
from pytorch_msssim import ms_ssim


class EncoderDecoder:
    """NOTE: virtual class that *you* need to subclass

    An instance of this class encodes an image, calculates the number of bits and decodes the encoded image, resulting in "transformed" image.

    Transformed image is similar to the original image, while the encoding+decoding process might have introduced some distortion.

    The instance may (say, H266 video encoder+decoder) or may not (say, jpeg encoder+decoder) have an internal state.
    """

    # helpers
    def compute_psnr(self, a, b):
        mse = torch.mean((a - b)**2).item()
        return -10 * math.log10(mse)

    def compute_msssim(self, a, b):
        return ms_ssim(a, b, data_range=1.).item()


    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.reset()
        raise (AssertionError("virtual"))

    def reset(self):
        """Reset the internal state of the encoder & decoder, if there is any"""
        self.cc = 0

    def __call__(self, x) -> tuple:
        """Push images(s) through the encoder+decoder, returns number of bits for each image and encoded+decoded images

        :param x: a FloatTensor with dimensions (batch, channels, y, x)

        Returns (nbitslist, x_hat), where nbitslist is a list of number of bits and x_hat is the image that has gone throught the encoder/decoder process
        """
        self.cc += 1
        raise (AssertionError("virtual"))
        return None, None

    def BGR(self, bgr_image, tag=None):
        """
        :param bgr_image: numpy BGR image (y,x,3)
        :param tag: a string that can be used to identify & cache images (optional)

        Takes in an BGR image, pushes it through encoder + decoder.

        Returns nbits, transformed BGR image.
        """
        raise (AssertionError("virtual"))


class VoidEncoderDecoder(EncoderDecoder):
    """Does no encoding/decoding whatsoever.  Use for debugging."""

    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.reset()

    def reset(self):
        """Reset the internal state of the encoder & decoder, if any"""
        self.cc = 0

    def __call__(self, x) -> tuple:
        """Push images(s) through the encoder+decoder, returns bbps and encoded+decoded images

        :param x: a FloatTensor with dimensions (batch, channels, y, x)

        Returns (nbitslist, x_hat), where nbitslist is a list of number of bits and x_hat is the image that has gone throught the encoder/decoder process
        """
        self.cc += 1
        return [0], x

    def BGR(self, bgr_image, tag=None):
        """
        :param bgr_image: numpy BGR image (y,x,3)
        :param tag: a string that can be used to identify & cache images (optional)

        Returns BGR image that has gone through transformation (the encoding + decoding process)

        Returns nbits, transformed BGR image
        """
        return 0, bgr_image
