# Copyright (c) 2021-2022, InterDigital Communications, Inc
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

import os
import warnings

from typing import Dict

import torch
import torch.nn as nn
from torch import Tensor

from compressai.layers import GDN
from compressai.models.google import CompressionModel
from compressai.models.utils import conv, deconv

from compressai_vision.evaluation.pipeline import CompressAIEncoderDecoder


class FactorizedPrior(CompressionModel):
    r"""Factorized Prior model from J. Balle, D. Minnen, S. Singh, S.J. Hwang,
    N. Johnston: `"Variational Image Compression with a Scale Hyperprior"
    <https://arxiv.org/abs/1802.01436>`_, Int Conf. on Learning Representations
    (ICLR), 2018.

    Args:
        N (int): Number of channels
        M (int): Number of channels in the expansion layers (last layer of the
            encoder and last layer of the hyperprior decoder)
    """

    def __init__(self, N, M, **kwargs):
        super().__init__(entropy_bottleneck_channels=M, **kwargs)

        self.g_a = nn.Sequential(
            conv(3, N),
            GDN(N),
            conv(N, N),
            GDN(N),
            conv(N, N),
            GDN(N),
            conv(N, M),
        )

        self.g_s = nn.Sequential(
            deconv(M, N),
            GDN(N, inverse=True),
            deconv(N, N),
            GDN(N, inverse=True),
            deconv(N, N),
            GDN(N, inverse=True),
            deconv(N, 3),
        )

        self.N = N
        self.M = M

    @property
    def downsampling_factor(self) -> int:
        return 2**4

    def forward(self, x):
        y = self.g_a(x)
        y_hat, y_likelihoods = self.entropy_bottleneck(y)
        x_hat = self.g_s(y_hat)

        return {
            "x_hat": x_hat,
            "likelihoods": {
                "y": y_likelihoods,
            },
        }

    @classmethod
    def from_state_dict(cls, state_dict):
        """Return a new model instance from `state_dict`."""
        N = state_dict["g_a.0.weight"].size(0)
        M = state_dict["g_a.6.weight"].size(0)
        net = cls(N, M)
        net.load_state_dict(state_dict)
        return net

    def compress(self, x):
        # x: (batch, 3, H, W)
        y = self.g_a(x) # (batch, FM, FM-H, FM-W)
        y_strings = self.entropy_bottleneck.compress(y) # list: first element is bytes
        return {"strings": [y_strings], "shape": y.size()[-2:]}
        # --> res["strings"][0][0] has the bytes, shape has the FM dimensions

    def decompress(self, strings, shape):
        assert isinstance(strings, list) and len(strings) == 1
        y_hat = self.entropy_bottleneck.decompress(strings[0], shape)
        x_hat = self.g_s(y_hat).clamp_(0, 1) # (batch, 3, H W)
        return {"x_hat": x_hat}


def rename_key(key: str) -> str:
    """Rename state_dict key."""

    # Deal with modules trained with DataParallel
    if key.startswith("module."):
        key = key[7:]

    # ResidualBlockWithStride: 'downsample' -> 'skip'
    if ".downsample." in key:
        return key.replace("downsample", "skip")

    # EntropyBottleneck: nn.ParameterList to nn.Parameters
    if key.startswith("entropy_bottleneck."):
        if key.startswith("entropy_bottleneck._biases."):
            return f"entropy_bottleneck._bias{key[-1]}"

        if key.startswith("entropy_bottleneck._matrices."):
            return f"entropy_bottleneck._matrix{key[-1]}"

        if key.startswith("entropy_bottleneck._factors."):
            return f"entropy_bottleneck._factor{key[-1]}"

    return key


def load_state_dict(state_dict: Dict[str, Tensor]) -> Dict[str, Tensor]:
    """Convert state_dict keys."""
    state_dict = {rename_key(k): v for k, v in state_dict.items()}
    return state_dict


def getEncoderDecoder(quality=None, device="cpu", scale=None, ffmpeg="ffmpeg", dump=False, **kwargs):
    """Returns CompressAIEncoderDecoder instance

    - Maps quality parameters to checkpoint files
    - Loads the model
    - Returns the EncoderDecoder instance
    """
    assert(quality is not None), "please provide a quality parameters"

    for key, value in kwargs.items():
        print("WARNING: unused parameter", key, "with value", value)

    qpoint_per_file = {
        1 : "bmshj2018-factorized-prior-1-446d5c7f.pth.tar",
        2 : "bmshj2018-factorized-prior-2-87279a02.pth.tar"
    }

    try:
        cp_file = qpoint_per_file[quality]
    except KeyError:
        print("Invalid quality point", quality)
        raise

    # path of _this_ directory:
    pwd = os.path.dirname(__file__)
    # correct filename paths:
    cp_file = os.path.join(pwd, cp_file)

    # instantiate the model:
    net = FactorizedPrior(128, 192)

    try:
        checkpoint = torch.load(cp_file)
        if "network" in checkpoint:
            state_dict = checkpoint["network"]
        elif "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        else:
            state_dict = checkpoint

        state_dict = load_state_dict(state_dict)
        # For now, update is forced in case
        net = net.from_state_dict(state_dict)
        net.update(force=True)
        # net = net.to(device).eval() # done by the main program
    except Exception as e:
        print("\nLoading checkpoint failed!\n")
        raise e

    """CompressAIEncoderDecoder knows how to handle standard CompressAI models.  It uses
    the compress and decompress methods (see above)
    """
    enc_dec = CompressAIEncoderDecoder(
        net, device=device, scale=scale, 
        ffmpeg=ffmpeg, 
        dump=dump
    )
    return enc_dec


