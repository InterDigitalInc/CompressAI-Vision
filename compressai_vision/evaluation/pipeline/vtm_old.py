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
import math
import os
import subprocess

import numpy as np
import torch

from PIL import Image

# from regex import B
from torchvision import transforms

from .base import EncoderDecoder


class VTMEncoderDecoder(EncoderDecoder):
    """EncoderDecoder class for VTM encoder

    :param encoderApp: path of VTM encoder
    :param decoderApp: path of VTM decoder
    :param vtm_cfg: path of encoder cfg
    :param qp: the default quantization parameter of the instance. Integer from 0 to 63
    :param save_transformed: option to save intermedidate images. default = False
    """

    def __init__(self, encoderApp, decoderApp, vtm_cfg, qp, save_transformed=False):
        self.logger = logging.getLogger(self.__class__.__name__)

        if os.path.isfile(vtm_cfg):
            self.vtm_cfg = vtm_cfg
        if os.path.isfile(encoderApp):
            self.encoderApp = encoderApp
        else:
            self.logger.critical("VTM encoder not found at %s", encoderApp)
        if os.path.isfile(decoderApp):
            self.decoderApp = decoderApp
        else:
            self.logger.critical("VTM decoder not found at %s", decoderApp)
        self.tmp_output_folder = "/dev/shm/"
        self.device = "cpu"
        self.qp = qp
        self.save_transformed = save_transformed
        self.reset()

    def reset(self):
        """Reset encoder/decoder internal state? Jacky: TODO."""
        super().reset()
        self.imcount = 0

    def process_cmd(self, cmd, print_out=False):
        """
        process bash cmd
        :param cmd: bash command
        :param print_out: show printout. Default: False
        """

        if print_out:
            print(cmd)
        p = subprocess.Popen(
            cmd, shell=True, stderr=subprocess.STDOUT, stdout=subprocess.PIPE
        )

        for line in p.stdout.readlines():
            if print_out:
                self.logger.debug(line)
                print(line)

        if print_out:
            print("Done")

    def __encode_ffmpeg__(self, x, qp, bin_path, bPrint=True):

        """
        Encode input image x with VTM encoder. Padding is done with ffmpeg if the image has singular dimenions.

        :param x: a FloatTensor with dimensions (batch, channels, y, x)
        :param qp: quantization parameter. Integer: 0-63
        :param bin_path: path of output bitstream

        """
        x = torch.squeeze(x, 0)
        tensor_size = x.size()

        image_name = "tmp.jpg"

        width = tensor_size[-1]
        height = tensor_size[-2]
        padded_hgt = math.ceil(height / 2) * 2
        padded_wdt = math.ceil(width / 2) * 2

        tmp_jpg_path = os.path.join(self.tmp_output_folder, image_name)
        # function to save x as tmp_jpg
        img = transforms.ToPILImage()(x).convert("RGB")
        img.save(tmp_jpg_path)

        yuv_image_path = os.path.join(
            self.tmp_output_folder, image_name.replace(".jpg", ".yuv")
        )
        temp_yuv_path = os.path.join(self.tmp_output_folder, "rec.yuv")
        # 1. use ffmpeg to rescale and pad the input image, and then convert to yuv
        cmd_str = f'ffmpeg -y -i {tmp_jpg_path} -vf "pad=ceil(iw/2)*2:ceil(ih/2)*2" -f rawvideo -pix_fmt yuv420p -dst_range 1 {yuv_image_path}'
        self.process_cmd(cmd_str, print_out=bPrint)

        # use VTM encoder to generate bitstream
        cmd_str = f"{self.encoderApp} -c {self.vtm_cfg} -i {yuv_image_path} -b {bin_path} -o {temp_yuv_path} -fr 1 -f 1 -wdt {padded_wdt} -hgt {padded_hgt} -q {qp}"
        self.process_cmd(cmd_str, print_out=bPrint)

        # check if bin is generated correctly

        check_bin = os.path.isfile(bin_path)
        if check_bin:
            self.logger.debug("Bitstream stored at: %s", bin_path)
        else:
            self.logger.critical("Bitstream %s storage failed.", bin_path)

    def __decode_ffmpeg__(self, bin_path, width, height, bPrint=True):

        """
        Decode the bitstream with VTM decoder and return recontructed image x_hat. Cropping is done with ffmpeg.
        :param bin_path: Input bitstream path
        :param width: The width of the reconstructed image
        :param height: The height of the reconstructed image
        """

        padded_hgt = math.ceil(height / 2) * 2
        padded_wdt = math.ceil(width / 2) * 2
        rec_yuv_path = os.path.join(
            self.tmp_output_folder, bin_path.replace(".bin", ".yuv")
        )
        rec_png_path = os.path.join(
            self.tmp_output_folder, bin_path.replace(".bin", "_tmp.png")
        )
        output_image_name = os.path.join(
            self.tmp_output_folder, bin_path.replace(".bin", ".png")
        )

        # use VTM decoder
        cmd_str = f"{self.decoderApp} -b {bin_path} -o {rec_yuv_path}"
        self.process_cmd(cmd_str, print_out=bPrint)

        # use ffmpeg to convert yuv back to png
        cmd_str = f"ffmpeg -y -f rawvideo -pix_fmt yuv420p10le -s {padded_wdt}x{padded_hgt} -src_range 1 -i {rec_yuv_path} -frames 1  -pix_fmt rgb24 {rec_png_path}"
        self.process_cmd(cmd_str, print_out=bPrint)

        cmd_str = f'ffmpeg -y -i {rec_png_path} -vf "crop={width}:{height}" {output_image_name}'
        self.process_cmd(cmd_str, print_out=bPrint)

        img = Image.open(output_image_name)
        x_hat = transforms.ToTensor()(img).unsqueeze(0)

        return x_hat

    def __call_ffmpeg__(self, x, qp):
        """Push images(s) through the encoder+decoder, returns bbps and encoded+decoded images

        :param x: a FloatTensor with dimensions (batch, channels, y, x)

        :param qp: quantization parameter. Integer: 0-63

        Returns (bpps, x_hat), where x_hat is batch of images that have gone through the encoder/decoder process,
        bpps is a list of bits per pixel of each compressed image in that batch

        Rescaling and colour transformation are done with ffmpeg.
        """

        bin_path = "tmp.bin"
        tensor_size = x.size()
        width = tensor_size[-1]
        height = tensor_size[-2]
        self.__encode_ffmpeg__(x, qp, bin_path, bPrint=False)
        bpp = os.path.getsize(bin_path)
        img = self.__decode_ffmpeg__(bin_path, width, height, bPrint=False)
        return [bpp], img

    def __call__(self, x, qp=None):

        """Push images(s) through the encoder+decoder, returns bbps and encoded+decoded images

        :param x: a FloatTensor with dimensions (batch, channels, y, x)
        :param qp: quantization parameter. Default: None- use the default qp of the instance.

        Returns (bpps, x_hat), where x_hat is batch of images that have gone through the encoder/decoder process,
        bpps is a list of bits per pixel of each compressed image in that batch

        currently this method is implemented in self.__call_ffmpeg__, rescaling and colour transformation are done with ffmpeg.
        """

        if qp is None:
            qp = self.qp
        return self.__call_ffmpeg__(x, qp)

    def BGR(self, bgr_image):
        """
        :param bgr_image: numpy BGR image (y,x,3)

        Returns BGR image that has gone through VTM encode/decode.  Jacky: TODO
        """
        rgb_image = bgr_image[:, :, [2, 1, 0]]  # BGR --> RGB
        # rgb_image (y,x,3) to FloatTensor (1,3,y,x):
        # TODO: need to do padding here?
        x = transforms.ToTensor()(rgb_image).unsqueeze(0)
        x = x.to(self.device)
        bpp, x_hat = self(x)
        x_hat = x_hat.squeeze(0).to("cpu")
        rgb_image_hat = np.array(transforms.ToPILImage()(x_hat))
        bgr_image_hat = rgb_image_hat[:, :, [2, 1, 0]]  # RGB --> BGR
        # TODO: need to remove padding / resize here?

        if self.save_transformed:
            try:
                os.mkdir("compressai_encoder_decoder")
            except FileExistsError:
                pass
            Image.fromarray(
                bgr_image_hat[:, :, ::-1]
                # bgr_image
            ).save(
                os.path.join(
                    "compressai_encoder_decoder", "dump_" + str(self.imcount) + ".png"
                )
            )
        self.logger.debug(
            "input & output sizes: %s %s. bps = %s",
            bgr_image.shape,
            bgr_image_hat.shape,
            bpp[0],
        )
        # print(">> cc, bpp_sum ", self.cc, self.bpp_sum)
        return bpp[0], bgr_image_hat
