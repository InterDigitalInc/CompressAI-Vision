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

import logging
import os

import numpy as np
import torch

from torchvision import transforms

from compressai_vision.constant import vf_per_scale, inv_vf_per_scale
from compressai_vision.ffmpeg import FFMpeg
from compressai_vision.tools import dumpImageArray, test_command

from .base import EncoderDecoder

torch.backends.cudnn.deterministic = True
torch.set_num_threads(1)


class CompressAIEncoderDecoder(EncoderDecoder):
    """EncoderDecoder class for CompressAI

    :param net: compressai network, for example:

    ::

        net = bmshj2018_factorized(quality=2, pretrained=True).eval().to(device)

    :param device: "cpu" or "cuda"
    :param dump: (debugging) dump transformed images to disk.  default = False
    :param m: images should be multiples of this number.  If not, a padding is applied before passing to compressai.  default = 64
    :param ffmpeg: ffmpeg command used for padding/scaling (as defined by VCM working group). Default: "ffmpeg".
    :param scale: enable the VCM working group defined padding/scaling pre & post-processings steps.
                  Possible values: 100 (default), 75, 50, 25.  Special value: None = ffmpeg scaling.  100 equals to a simple padding operation
    :param dump: debugging option: dump input, intermediate and output images to disk in local directory

    This class uses CompressAI model API's ``compress`` and ``decompress`` methods, so if your model has them, then it is
    compatible with this particular ``EncoderDecoder`` class, in detail:

    ::
        # CompressAI model API:
        # compression:
        out_enc = self.net.compress(x)
        bitstream = out_enc["strings"][0][0]  # compressed bitstream
        # decompression:
        out_dec = self.net.decompress(out_enc["strings"], out_enc["shape"])
        x_hat = out_dec["x_hat"] # reconstructed image

    """

    toFloat = transforms.ConvertImageDtype(torch.float)
    toByte = transforms.ConvertImageDtype(torch.uint8)

    def __init__(
        self,
        net,
        device="cpu",
        dump=False,
        m: int = 64,
        ffmpeg="ffmpeg",
        scale: int = None,
        half=False,
    ):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.net = net
        self.device = device
        self.dump = dump
        self.m = 64
        self.reset()
        self.save_folder = "compressai_encoder_decoder"
        if self.dump:
            self.logger.info("Will save images to folder %s", self.save_folder)
            os.makedirs(self.save_folder, exist_ok=True)

        self.scale = scale
        if self.scale is not None:
            assert self.scale in vf_per_scale.keys(), "incorrect scaling constant"
        try:
            self.ffmpeg_comm = test_command(ffmpeg)
        except FileNotFoundError:
            raise (AssertionError("cant find ffmpeg"))
        self.ffmpeg = FFMpeg(self.ffmpeg_comm, self.logger)
        self.compute_metrics = True
        self.half = half

    # some parameters can also be set after ctor
    def computeMetrics(self, state: bool):
        self.compute_metrics = state

    def reset(self):
        """Reset internal image counter"""
        super().reset()
        self.imcount = 0
        self.latest_psnr = None
        self.latest_msssim = None

    def __call__(self, x):
        """Push images(s) through the encoder+decoder, returns nbitslist (list of number of bits) and encoded+decoded images

        :param x: a FloatTensor with dimensions (batch, channels, y, x)

        WARNING: we assume that batch=1

        Returns (nbitslist, x_hat), where x_hat is batch of images that have gone through the encoder/decoder process,
        nbitslist is a list of number of bits of each compressed image in that batch
        """
        assert x.size()[0] == 1, "batch dimension must be 1"
        if self.half:
            x = x.half()

        with torch.no_grad():
            # compression
            out_enc = self.net.compress(x)
            # decompression
            out_dec = self.net.decompress(out_enc["strings"], out_enc["shape"])

        # TODO: out_enc["strings"][batch_index?][what?] .. for batch sizes > 1
        total_strings = 0
        for bitstream in out_enc["strings"]:
            total_strings += len(bitstream[0])
        # print("bitstream is>", type(bitstream))
        # print("x_hat is>", x_hat.shape)
        # num_pixels = x.shape[2] * x.shape[3]
        # print("num_pixels", num_pixels)
        nbits = 8 * total_strings  # BITS not BYTES
        nbitslist = [nbits]
        x_hat = (
            torch.round(out_dec["x_hat"].clamp(0, 1) * 255.0) / 255.0
        )  # (batch, 3, H, W)  # reconstructed image

        if self.compute_metrics:
            self.latest_psnr = self.compute_psnr(x, x_hat)
            self.latest_msssim = self.compute_msssim(x, x_hat)
        return nbitslist, x_hat

    def getMetrics(self):
        return self.latest_psnr, self.latest_msssim

    def BGR(self, bgr_image: np.array, tag=None) -> tuple:
        """Return transformed image and nbits for a BGR image

        :param bgr_image: numpy BGR image (y,x,3)
        :param tag: a string that can be used to identify & cache images (optional)

        Returns number of bits and transformed BGR image that has gone through compressai encoding+decoding.

        - Scales the image if scaling is requested (1) [with ffmpeg]
        - Pads the image for CompressAI (2) [with ffmpeg - feel free to switch to torch if you want]
        - Runs the image through CompressAI model
        - Removes padding (2) [with ffmpeg]
        - Backscales (1) [with ffmpeg]

        Necessary padding for compressai is added and removed on-the-fly
        """
        # TO RGB & TENSOR
        rgb_image = bgr_image[:, :, [2, 1, 0]]  # BGR --> RGB

        """
        rgb_image       original img
        scaled          scaled (if requested) (1)
        padded          padded for compressai (2)
        padded_hat      encoded & decoded with compressai
        scaled_hat      padding removed (2)
        rgb_image_hat   scaling removed (1)
        """

        tag_ = tag if tag else str(self.imcount)

        if self.dump:
            dumpImageArray(rgb_image, self.save_folder, "original_" + tag_ + ".png")

        do_scaling = (self.scale is not None) and self.scale != 100

        if do_scaling:
            # the padding for compressai is bigger than this one, so it is innecessary to do this
            # on the other hand, if we want to play strictly by the VCM working group book, then
            # this should be done..?
            #
            # 1. MPEG-VCM: ffmpeg -i {input_jpg_path} -vf “pad=ceil(iw/2)*2:ceil(ih/2)*2” {input_tmp_path}
            #
            vf = vf_per_scale[self.scale]
            scaled = self.ffmpeg.ff_op(rgb_image, vf)
        else:
            scaled = rgb_image

        if self.dump:
            dumpImageArray(scaled, self.save_folder, "scaled_" + tag_ + ".png")

        # *** Add padding for CompressAI ***
        # https://ffmpeg.org/ffmpeg-filters.html#Examples-100
        pad_vf = "pad=ceil(iw/{S})*{S}:ceil(ih/{S})*{S}".format(S=self.m)
        padded = self.ffmpeg.ff_op(scaled, pad_vf)
        if self.dump:
            dumpImageArray(padded, self.save_folder, "padded_" + tag_ + ".png")

        # print(">orig dims", scaled.shape)
        # print(">padded dims", padded.shape)
        x_pad = transforms.ToTensor()(padded).unsqueeze(0)

        # RUN COMPRESSAI
        x_pad = x_pad.to(self.device)
        nbitslist, x_hat_pad = self(x_pad)
        x_hat_pad = x_hat_pad.to("cpu")

        # TO NUMPY ARRAY & BGR IMAGE
        x_hat_pad = x_hat_pad.squeeze(0)
        padded_hat = np.array(transforms.ToPILImage()(x_hat_pad))
        if self.dump:
            dumpImageArray(padded_hat, self.save_folder, "padded_hat_" + tag_ + ".png")

        # *** Remove CompressAI padding ***
        scaled_hat = (
            self.ffmpeg.ff_op(  # https://ffmpeg.org/ffmpeg-filters.html#Examples-60
                padded_hat,
                "crop={width}:{height}:0:0".format(
                    width=scaled.shape[1], height=scaled.shape[0]
                ),
            )
        )
        if self.dump:
            dumpImageArray(scaled_hat, self.save_folder, "scaled_hat_" + tag_ + ".png")

        # *** Remove scaling ***
        if do_scaling:
            # was scaled, so need to backscale
            vf = inv_vf_per_scale[self.scale]
            rgb_image_hat = self.ffmpeg.ff_op(
                scaled_hat,
                vf.format(width=rgb_image.shape[1], height=rgb_image.shape[0]),
            )
        else:
            rgb_image_hat = scaled_hat

        # SAVE IMAGE IF
        if self.dump:
            dumpImageArray(
                rgb_image_hat, self.save_folder, "rgb_image_hat_" + tag_ + ".png"
            )

        bgr_image_hat = rgb_image_hat[:, :, [2, 1, 0]]  # RGB --> BGR

        self.logger.debug(
            "input & output sizes: %s %s. nbits = %s",
            bgr_image.shape,
            bgr_image_hat.shape,
            nbitslist[0],
        )
        # print(">> cc, bpp_sum ", self.cc, self.bpp_sum)
        self.imcount += 1
        return nbitslist[0], bgr_image_hat
