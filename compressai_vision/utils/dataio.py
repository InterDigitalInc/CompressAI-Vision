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

import enum
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
import torch.nn.functional as F
import yuvio
from torch import Tensor

from .logger import warning


class PixelFormat(enum.Enum):
    """assigned strings are compatiable with YUVIO library pixel format"""

    YUV400 = ("gray", 8)  # planar 4:0:0 YUV 8-bit
    YUV400_10le = ("gray10le", 10)  # planar 4:0:0 YUV 10-bit

    # Not supported yet
    # (TODO: (choih) Support other formats)
    # YUV420      = ("yuv420p", 8)      # planar 4:2:0 YUV 8-bit
    # YUV420_10le = ('yuv420p10le', 10) # planar 4:2:0 YUV 10-bit
    # YUV422      = ("yuv422p", 8)      # planar 4:2:2 YUV 8-bit
    # YUV422_10le = ('yuv422p10le', 10) # planar 4:2:2 YUV 10-bit
    # YUV444      = ("yuv444p", 8)      # planar 4:4:4 YUV 8-bit
    # YUV444_10le = ('yuv444p10le', 10) # planar 4:4:4 YUV 10-bit
    # RGB         = ("yuv444p", 8)      # planar 4:4:4 RGB 8-bit


bitdepth_to_dtype = {
    8: np.uint8,
    10: np.uint16,
    12: np.uint16,
    14: np.uint16,
    16: np.uint16,
}

bitdepth_to_mid_level = {
    8: 128,
    10: 512,
    12: 2048,
    14: 8192,
    16: 32768,
}


class readwriteYUV:
    """ " """

    def __init__(
        self, device, format: PixelFormat = PixelFormat.YUV400, align=2, surround=False
    ):
        """ """
        self._device = device
        self._format = format
        self._align = align
        self._surround = surround

    @property
    def device(self):
        return self._device

    @property
    def format(self):
        return self._format

    @property
    def bitdepth(self):
        return self._format[1]

    @property
    def resolution_multiple_of_(self):
        return self._align

    @staticmethod
    def _compute_new_frame_resolution(frmWidth, frmHeight, align):
        # recalculate frame resolution with alignment factor for to-be-saved YUVs
        H = (frmHeight + align - 1) // align * align
        W = (frmWidth + align - 1) // align * align
        return W, H

    @staticmethod
    def _path_check(path):
        if not Path(path).exists() or not Path(path).is_file():
            raise RuntimeError(f'Invalid file "{path}"')
    @staticmethod
    def _path_create(path):
        wp = Path(path).parent
        wp.mkdir(parents=True, exist_ok=True)
        if not wp.exists() or not wp.is_dir():
            raise RuntimeError(f'Invalid file "{wp}"')

    @staticmethod
    def pad(x, p=2, mid=0, surround=False):
        h, w = x.size(1), x.size(2)
        H = (h + p - 1) // p * p
        W = (w + p - 1) // p * p

        if surround is True:
            padding_left = (W - w) // 2
            padding_top = (H - h) // 2
        else:
            padding_left = padding_top = 0
        padding_right = W - w - padding_left
        padding_bottom = H - h - padding_top

        return F.pad(
            x,
            (padding_left, padding_right, padding_top, padding_bottom),
            mode="constant",
            value=mid,
        )

    @staticmethod
    def crop(x, size, surround=False):
        gap_in_width, gap_in_height = size

        if surround is True:
            padding_left = gap_in_width // 2
            padding_top = gap_in_height // 2
        else:
            padding_left = padding_top = 0

        padding_right = gap_in_width - padding_left
        padding_bottom = gap_in_height - padding_top

        return F.pad(
            x,
            (-padding_left, -padding_right, -padding_top, -padding_bottom),
            mode="constant",
            value=0,
        )

    def setReader(
        self,
        read_path: str,
        frmWidth,
        frmHeight,
        format=None,
        align=None,
        surround=None,
    ):
        """ """

        format = self._format if format is None else format
        align = self._align if align is None else align
        surround = self._surround if surround is None else surround

        self._path_check(read_path)
        _frmWidth, _frmHeight = self._compute_new_frame_resolution(
            frmWidth, frmHeight, align
        )

        # save resolution difference in width and height
        # between the original frame and padded frame
        self._gap_in_width = _frmWidth - frmWidth
        self._gap_in_height = _frmHeight - frmHeight

        self.reader = yuvio.get_reader(
            read_path, _frmWidth, _frmHeight, format.value[0]
        )
        self.pixel_bitdepth = format.value[1]

    def setWriter(
        self,
        write_path: str,
        frmWidth,
        frmHeight,
        format=None,
        align=None,
        surround=None,
    ):
        """ """

        format = self._format if format is None else format
        align = self._align if align is None else align
        surround = self._surround if surround is None else surround

        self._path_create(write_path)
        frmWidth, frmHeight = self._compute_new_frame_resolution(
            frmWidth, frmHeight, align
        )
        self.writer = yuvio.get_writer(write_path, frmWidth, frmHeight, format.value[0])
        self.pixel_bitdepth = format.value[1]

    def write_single_frame(self, frame: Tensor, mid_level=None):
        if self.writer is None:
            raise RuntimeError("Please first setup the writer")

        assert (
            frame.dim() >= 2 and frame.dim() <= 4
        ), "Dimension of the input frame tensor shall be greater than 1 and less than 5"

        if frame.dim() == 4:
            if frame.size(0) > 1:
                warning(
                    "Size of input tensor at 0-th dimension is greater than 1. Only the first at 0-th dimension is valid"
                )
            frame = frame[0, ::]

        if frame.dim() == 3:
            if frame.size(0) > 3:
                warning(
                    "Number of color channels is greater than 3. Only the first three color components are valid"
                )
                frame = frame[0:3, ::]

        if frame.dim() == 2:
            assert (
                self.format == PixelFormat.YUV400
                or self.format == PixelFormat.YUV400_10le
            ), f"Input Dimension mismatches with format {self.format}"
            frame = frame.unsqueeze(0)

        dtype = bitdepth_to_dtype[self.pixel_bitdepth]

        if mid_level is None:
            mid_level = bitdepth_to_mid_level[self.pixel_bitdepth]

        frame = self.pad(frame, self._align, mid_level, surround=self._surround)

        if self.format == PixelFormat.YUV400 or self.format == PixelFormat.YUV400_10le:
            y_channel = np.array(frame[0].numpy(force=True), dtype=dtype)
            frame = yuvio.frame((y_channel, None, None), self._format.value[0])
        else:
            # TODO: (choih) Support other formats
            # components = []
            # for c in frame:
            #    channel = np.array(c.numpy(force=True), dtype=dtype)
            #    channel = channel.swapaxes(0, 1)
            raise NotImplementedError

        self.writer.write(frame)

    def write_multiple_frames(self, frames: Tensor, alignment=2):
        raise NotImplementedError
    def read_single_frame(self, frm_idx=0):
        """
        arguments:

        """
        frame = self.reader.read(index=frm_idx, count=1)[0]
        y, u, v = frame.split()

        if self.format == PixelFormat.YUV400 or self.format == PixelFormat.YUV400_10le:
            assert u == v is None
            out = torch.from_numpy(y.astype("float32")).to(self._device)
            out = self.crop(
                out, (self._gap_in_width, self._gap_in_height), self._surround
            )
        else:
            # TODO: (choih) Support other formats
            # components = []
            # for c in frame:
            #    channel = np.array(c.numpy(force=True), dtype=dtype)
            #    channel = channel.swapaxes(0, 1)
            raise NotImplementedError

        return out

    def read_multiple_frames(self, crop: Tuple):
        raise NotImplementedError


# y = 255 * np.ones((1920, 1080), dtype=np.uint8)
# u = np.zeros((960, 540), dtype=np.uint8)
# v = np.zeros((960, 540), dtype=np.uint8)
# frame_420 = yuvio.frame((y, u, v), "yuv420p")
# frame_400 = yuvio.frame((y, None, None), "gray")

# for yuv_frame in reader:
#    writer.write(yuv_frame)

# yuv_frame = yuvio.imread("example_yuv420p.yuv", 1920, 1080, "yuv420p")
# yuvio.imwrite("example_yuv420p_copy.yuv", yuv_frame)

# yuv_frames = yuvio.mimread("example_yuv420p.yuv", 1920, 1080, "yuv420p")
# yuvio.mimwrite("example_yuv420p_copy.yuv", yuv_frames)