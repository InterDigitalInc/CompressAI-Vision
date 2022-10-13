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

import io
import shlex
import subprocess

import numpy as np

from PIL import Image


class FFMpeg:
    """FFMpeg encapsulation

    :param ffmpeg: the ffmpeg command
    :param logger: a logger instance

    TODO: for reading video, define video input, keep ffmpeg process alive & stream into stdin
    """

    def __init__(self, ffmpeg, logger):
        self.ffmpeg = ffmpeg
        self.logger = logger

    def ff_op(self, rgb_image: np.array, op) -> np.array:
        """takes as an input a numpy RGB array (y,x,3)

        Outputs numpy RGB array after certain transformation
        """
        pil_img = Image.fromarray(rgb_image)
        f = io.BytesIO()
        pil_img.save(f, format="png")

        comm = '{ffmpeg} -y -hide_banner -loglevel error -i pipe: -vf "{op}" -f apng pipe:'.format(
            ffmpeg=self.ffmpeg, op=op
        )
        self.logger.debug(comm)
        args = shlex.split(comm)
        p = subprocess.Popen(
            args, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )
        stdout, stderr = p.communicate(f.getvalue())
        if (stdout is None) or (len(stdout) < 5) or p.returncode != 0:
            # print(stderr.decode("utf-8"))
            self.logger.fatal("ffmpeg failed with %s", stderr.decode("utf-8"))
            return None
        f2 = io.BytesIO(stdout)
        pil_img2 = Image.open(f2).convert("RGB")
        return np.array(pil_img2)

    def ff_RGB24ToRAW(self, rgb_image: np.array, form) -> bytes:
        """takes as an input a numpy RGB array (y,x,3)

        ffmpeg -i input.png -f rawvideo -pix_fmt yuv420p -dst_range 1 output.yuv

        produces raw video frame bytes in the given pixel format
        """
        pil_img = Image.fromarray(rgb_image)
        f = io.BytesIO()
        pil_img.save(f, format="png")

        comm = "{ffmpeg} -y -hide_banner -loglevel error -i pipe: -f rawvideo -pix_fmt {form} -dst_range 1 pipe:".format(
            ffmpeg=self.ffmpeg, form=form
        )
        self.logger.debug(comm)
        args = shlex.split(comm)
        p = subprocess.Popen(
            args, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )
        stdout, stderr = p.communicate(f.getvalue())
        if (stdout is None) or (len(stdout) < 5) or p.returncode != 0:  # say
            self.logger.fatal("ffmpeg failed with %s", stderr.decode("utf-8"))
            return None
        f2 = io.BytesIO(stdout)
        return f2.read()

    def ff_RAWToRGB24(self, raw: bytes, form, width=None, height=None) -> bytes:
        """takes as an input a numpy RGB array (y,x,3)

        ffmpeg -y -f rawvideo -pix_fmt yuv420p -s 768x512 -src_range 1 -i test.yuv -frames 1 -pix_fmt rgb24 out.png

        produces RGB image from raw video format
        """
        assert width is not None
        assert height is not None
        assert isinstance(raw, bytes)

        comm = "{ffmpeg} -y -hide_banner -loglevel error -f rawvideo -pix_fmt {form} -s {width}x{height} -src_range 1 -i pipe: -frames 1 -pix_fmt rgb24 -f apng pipe:".format(
            ffmpeg=self.ffmpeg, form=form, width=width, height=height
        )
        self.logger.debug(comm)
        args = shlex.split(comm)
        p = subprocess.Popen(
            args, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )
        stdout, stderr = p.communicate(raw)
        if (stdout is None) or (len(stdout) < 5) or p.returncode != 0:  # say
            self.logger.fatal("ffmpeg failed with %s", stderr.decode("utf-8"))
            return None
        f = io.BytesIO(stdout)
        pil_img = Image.open(f)
        return np.array(pil_img)
