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

import torch
from pytorch_msssim import ms_ssim


class ModelSplit:
    """NOTE: virtual class that *you* need to subclass

    An instance of this class encodes an image, calculates the number of bits and decodes the encoded image

    Transformed image is similar to the original image, while the encoding+decoding process might have introduced some distortion.

    The instance may (say, H266 video encoder+decoder) or may not (say, jpeg encoder+decoder) have an internal state.
    """

    def __init__(self):
        # Compression -
        # Inference model -
        self.logger = logging.getLogger(self.__class__.__name__)
        self.compute_metrics = True
        self.reset()
        raise (AssertionError("virtual"))

    # helpers
    def compute_psnr(self, a, b):
        mse = torch.mean((a - b) ** 2).item()
        return -10 * math.log10(mse)

    def compute_msssim(self, a, b):
        return ms_ssim(a, b, data_range=1.0).item()

    def computeMetrics(self, state: bool):
        self.compute_metrics = state

    def getMetrics(self):
        """returns tuple with (psnr, mssim) from latest encode+decode calculation"""
        return None, None

    def __call__(self, x) -> tuple:
        """Push images(s) through the encoder+decoder, returns number of bits for each image and encoded+decoded images

        #TODO (fracape) video
        :param x: a FloatTensor with dimensions (batch, channels, h, w)

        Returns (nbitslist, x_hat), where nbitslist is a list of number of bits and x_hat is the image that has gone throught the encoder/decoder process
        """
        raise (AssertionError("virtual"))
        return None, None

    def preprocess(self, x):
        """Preprocess input according to a specific rquirement for input to the first part of the network

        - Potentially resize the input before feeding it in to first part of the network.
        - Potentially filter
        """
        raise NotImplementedError

    def from_input_to_features(self, x):
        """run the hinput according to a specific rquirement for input to encoder"""
        raise NotImplementedError

    def compress_features(self, x):
        """
        Inputs: tensors of features
        Returns nbits, transformed features.
        """

        raise NotImplementedError

    def from_features_to_output(self, out_decs):
        """Postprocess of possibly encoded/decoded data for various tasks inlcuding for human viewing and machine analytics"""
        raise NotImplementedError
