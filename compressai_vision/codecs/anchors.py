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

import errno
import logging
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import torch.nn as nn

from compressai_vision.model_wrappers import BaseWrapper
from compressai_vision.registry import register_codec
from compressai_vision.utils.dataio import PixelFormat, readwriteYUV

from .utils import MIN_MAX_DATASET, min_max_inv_normalization, min_max_normalization


def get_filesize(filepath: Union[Path, str]) -> int:
    return Path(filepath).stat().st_size


def run_cmdline(cmdline: List[Any], logpath: Optional[Path] = None) -> None:
    cmdline = list(map(str, cmdline))
    print(f"--> Running: {' '.join(cmdline)}", file=sys.stderr)

    if logpath is None:
        out = subprocess.check_output(cmdline).decode()
        if out:
            print(out)
        return

    p = subprocess.Popen(cmdline, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    with logpath.open("w") as f:
        if p.stdout is not None:
            for bline in p.stdout:
                line = bline.decode()
                f.write(line)
    p.wait()


@register_codec("vtm")
class VTM(nn.Module):
    """Encoder/Decoder class for VVC - VTM reference software"""

    def __init__(
        self,
        vision_model: BaseWrapper,
        dataset_name: "str" = "",
        **kwargs,
    ):
        super().__init__()
        self.encoder_path = Path(f"{kwargs['codec_paths']['encoder_exe']}")
        self.decoder_path = Path(f"{kwargs['codec_paths']['decoder_exe']}")
        self.cfg_file = Path(kwargs["codec_paths"]["cfg_file"])

        for file_path in [self.encoder_path, self.decoder_path, self.cfg_file]:
            if not file_path.is_file():
                raise FileNotFoundError(
                    errno.ENOENT, os.strerror(errno.ENOENT), file_path
                )

        self.qp = kwargs["encoder_config"]["qp"]
        self.eval_encode = kwargs["eval_encode"]

        self.dump_yuv = kwargs["dump_yuv"]
        # TODO (fracape) hacky, create separate function with LUT
        self.dataset = dataset_name
        if "sfu" in dataset_name:
            self.dataset = "SFU"
        self.vision_model = vision_model
        self.bitstream_dir = Path(kwargs["bitstream_dir"])
        self.bitstream_name = kwargs["bitstream_name"]
        if not self.bitstream_dir.is_dir():
            self.bitstream_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir = Path(kwargs["log_dir"])
        if not self.log_dir.is_dir():
            self.log_dir.mkdir(parents=True, exist_ok=True)

        self.min_max_dataset = MIN_MAX_DATASET[self.dataset]

        self.yuvio = readwriteYUV(device="cpu", format=PixelFormat.YUV400_10le)

    # can be added to base class (if inherited) | Should we inherit from the base codec?
    @property
    def qp_value(self):
        return self.qp

    # can be added to base class (if inherited) | Should we inherit from the base codec?
    @property
    def eval_encode_type(self):
        return self.eval_encode

    def get_encode_cmd(
        self, inp_yuv_path: Path, qp: int, bitstream_path: Path, width: int, height: int
    ) -> List[Any]:
        cmd = [
            self.encoder_path,
            "-i",
            inp_yuv_path,
            "-c",
            self.cfg_file,
            "-q",
            qp,
            "-o",
            "/dev/null",
            "-b",
            bitstream_path,
            "-wdt",
            width,
            "-hgt",
            height,
            "-fr",
            "1",
            "-f",
            "1",
            "--InputChromaFormat=400",
            "--InputBitDepth=10",
            "--ConformanceWindowMode=1",  # needed?
        ]
        return list(map(str, cmd))

    def get_decode_cmd(self, yuv_dec_path: Path, bitstream_path: Path) -> List[Any]:
        cmd = [self.decoder_path, "-b", bitstream_path, "-o", yuv_dec_path, "-d", 10]
        return list(map(str, cmd))

    def encode(
        self,
        x: Dict,
        file_prefix: str = "",
    ) -> bool:
        if file_prefix == "":
            file_prefix = self.bitstream_name.split(".")[0]

        yuv_in_path = f"{self.dump_yuv['yuv_in_dir']}/{file_prefix}_in.yuv"
        bitstream_path = f"{self.bitstream_dir}/{file_prefix}.bin"
        logpath = Path(f"{self.log_dir}/{file_prefix}_enc.log")

        
        bitdepth = 10  # TODO (fracape) (add this as config)

        (
            frame,
            self.feature_size,
            self.subframe_heights,
        ) = self.vision_model.reshape_feature_pyramid_to_frame(
            x["data"], packing_all_in_one=True
        )
        minv, maxv = self.min_max_dataset
        frame, mid_level = min_max_normalization(frame, minv, maxv, bitdepth=bitdepth)

        self.frame_height, self.frame_width = frame.size()

        # TODO (fracape) setWriter in init?
        self.yuvio.setWriter(
            write_path=yuv_in_path,
            frmWidth=self.frame_width,
            frmHeight=self.frame_height,
        )
        self.yuvio.write_single_frame(frame, mid_level=mid_level)

        cmd = self.get_encode_cmd(
            yuv_in_path,
            height=frame.size(1),
            width=frame.size(0),
            qp=self.qp,
            bitstream_path=bitstream_path,
        )
        # self.logger.debug(cmd)

        start = time.time()
        run_cmdline(cmd, logpath=logpath)
        enc_time = time.time() - start
        # self.logger.debug(f"enc_time:{enc_time}")
        return {
            "bytes": [get_filesize(bitstream_path)],
            "bitstream": bitstream_path,
        }

    def decode(self, bitstream_path: Path = None, file_prefix: str = "") -> bool:
        assert Path(bitstream_path).is_file()

        if file_prefix == "":
            file_prefix = bitstream_path.stem

        yuv_dec_path = f"{self.dump_yuv['yuv_dec_dir']}/{file_prefix}_dec.yuv"
        cmd = self.get_decode_cmd(
            bitstream_path=bitstream_path, yuv_dec_path=yuv_dec_path
        )
        # self.logger.debug(cmd)
        logpath = Path(f"{self.log_dir}/{file_prefix}_dec.log")

        start = time.time()
        run_cmdline(cmd, logpath=logpath)
        dec_time = time.time() - start
        # self.logger.debug(f"dec_time:{dec_time}")

        # TODO (fracape) setReader in init?
        self.yuvio.setReader(
            read_path=yuv_dec_path,
            frmWidth=self.frame_width,
            frmHeight=self.frame_height,
        )

        # read yuv rec
        # TODO (racapef) video: manage frame indexes
        rec_yuv = self.yuvio.read_single_frame(0)

        minv, maxv = self.min_max_dataset
        rec_yuv = min_max_inv_normalization(rec_yuv, minv, maxv, bitdepth=10)

        # TODO (fracape) should feature sizes be part of bitstream?
        features = self.vision_model.reshape_frame_to_feature_pyramid(
            rec_yuv,
            self.feature_size,
            self.subframe_heights,
            packing_all_in_one=True,
        )

        # features = {
        #     "data": self.vision_model.reshape_frame_to_feature_pyramid(
        #         rec_yuv,
        #         self.feature_size,
        #         self.subframe_heights,
        #         packing_all_in_one=True,
        #         ),
        #     "input_size": [582, 1333],
        #     "org_input_size": [447, 1024],
        # }
        return features


@register_codec("hm")
class HM(VTM):
    """Encoder / Decoder class for HEVC - HM reference software"""

    def __init__(
        self,
        vision_model: BaseWrapper,
        dataset_name: "str" = "",
        **kwargs,
    ):
        super().__init__(vision_model, dataset_name, **kwargs)

    def get_encode_cmd(
        self, inp_yuv_path: Path, qp: int, bitstream_path: Path, width: int, height: int
    ) -> List[Any]:
        cmd = [
            self.encoder_path,
            "-i",
            inp_yuv_path,
            "-c",
            self.cfg_file,
            "-q",
            qp,
            "-o",
            "/dev/null",
            "-b",
            bitstream_path,
            "-wdt",
            width,
            "-hgt",
            height,
            "-fr",
            "1",
            "-f",
            "1",
            "--InputChromaFormat=400",
            "--InputBitDepth=10",
            "--ConformanceWindowMode=1",  # needed?
        ]
        return list(map(str, cmd))
