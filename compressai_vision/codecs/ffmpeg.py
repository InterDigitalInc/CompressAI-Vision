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
import json
import logging
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import torch
import torch.nn as nn

from compressai_vision.model_wrappers import BaseWrapper
from compressai_vision.registry import register_codec
from compressai_vision.utils.dataio import PixelFormat, readwriteYUV

from .encdec_utils import get_raw_video_file_info
from .utils import MIN_MAX_DATASET, min_max_inv_normalization, min_max_normalization


# TODO (fracape) ffmpeg codecs could inherit from HM/VTM
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


@register_codec("x264")
class x264(nn.Module):
    """Encoder/Decoder class for x265"""

    def __init__(
        self,
        vision_model: BaseWrapper,
        dataset_name: "str" = "",
        **kwargs,
    ):
        super().__init__()

        self.qp = kwargs["encoder_config"]["qp"]
        self.frame_rate = kwargs["encoder_config"]["frame_rate"]
        self.eval_encode = kwargs["eval_encode"]

        self.dump_yuv = kwargs["dump_yuv"]
        # TODO (fracape) hacky, create separate function with LUT
        self.dataset = dataset_name
        if "sfu" in dataset_name:
            self.datacatalog = "SFU"
        else:
            self.datacatalog = dataset_name
        self.vision_model = vision_model

        self.min_max_dataset = MIN_MAX_DATASET[self.datacatalog]

        self.yuvio = readwriteYUV(device="cpu", format=PixelFormat.YUV400_10le)

        # TODO (fracape) move to cfg
        self.preset = kwargs["encoder_config"]["preset"]
        self.tune = kwargs["encoder_config"]["tune"]

    # can be added to base class (if inherited) | Should we inherit from the base codec?
    @property
    def qp_value(self):
        return self.qp

    # can be added to base class (if inherited) | Should we inherit from the base codec?
    @property
    def eval_encode_type(self):
        return self.eval_encode

    def get_encode_cmd(
        self,
        inp_yuv_path: Path,
        qp: int,
        bitstream_path: Path,
        width: int,
        height: int,
        nbframes: int = 1,
        frmRate: int = 1,
    ) -> List[Any]:
        cmd = [
            "ffmpeg",
            "-y",
            "-s:v",
            f"{width}x{height}",
            "-i",
            inp_yuv_path,
            "-c:v",
            "h264",
            "-crf",
            qp,
            "-preset",
            self.preset,
            "-bf",
            0,
            "-tune",
            self.tune,
            "-pix_fmt",
            "gray10le",  # to be checked
            "-threads",
            "4",
            bitstream_path,
        ]
        return cmd

    def get_decode_cmd(self, bitstream_path: Path, yuv_dec_path: Path) -> List[Any]:
        cmd = [
            "ffmpeg",
            "-y",
            "-i",
            bitstream_path,
            "-pix_fmt",
            "yuv420p",
            yuv_dec_path,
        ]
        return cmd

    def encode(
        self,
        x: Dict,
        codec_output_dir,
        bitstream_name,
        file_prefix: str = "",
    ) -> bool:
        bitdepth = 10  # TODO (fracape) (add this as config)

        (
            frames,
            self.feature_size,
            self.subframe_heights,
        ) = self.vision_model.reshape_feature_pyramid_to_frame(
            x["data"], packing_all_in_one=True
        )
        minv, maxv = self.min_max_dataset
        frames, mid_level = min_max_normalization(frames, minv, maxv, bitdepth=bitdepth)

        nbframes, frame_height, frame_width = frames.size()
        frmRate = self.frame_rate if nbframes > 1 else 1

        if file_prefix == "":
            file_prefix = f"{codec_output_dir}/{bitstream_name}_{frame_width}x{frame_height}_{frmRate}fps_{bitdepth}bit_p400"

        yuv_in_path = f"{file_prefix}_input.yuv"
        bitstream_path = f"{file_prefix}.mp4"
        logpath = Path(f"{file_prefix}_enc.log")

        self.yuvio.setWriter(
            write_path=yuv_in_path,
            frmWidth=frame_width,
            frmHeight=frame_height,
        )

        for frame in frames:
            self.yuvio.write_single_frame(frame, mid_level=mid_level)

        cmd = self.get_encode_cmd(
            yuv_in_path,
            height=frame.size(1),
            width=frame.size(0),
            qp=self.qp,
            bitstream_path=bitstream_path,
            nbframes=nbframes,
            frmRate=frmRate,
        )
        # self.logger.debug(cmd)

        start = time.time()
        run_cmdline(cmd, logpath=logpath)
        enc_time = time.time() - start
        # self.logger.debug(f"enc_time:{enc_time}")

        # to be compatible with the pipelines
        # per frame bits can be collected by parsing enc log to be more accurate
        avg_bytes_per_frame = get_filesize(bitstream_path) / nbframes
        all_bytes_per_frame = [avg_bytes_per_frame] * nbframes

        return {
            "bytes": all_bytes_per_frame,
            "bitstream": bitstream_path,
        }

    def decode(
        self,
        bitstream_path: Path = None,
        codec_output_dir: str = "",
        file_prefix: str = "",
    ) -> bool:
        bitstream_path = Path(bitstream_path)
        assert bitstream_path.is_file()

        if file_prefix == "":
            file_prefix = bitstream_path.stem

        video_info = get_raw_video_file_info(file_prefix.split("qp")[-1])
        frame_width = video_info["width"]
        frame_height = video_info["height"]
        yuv_dec_path = f"{codec_output_dir}/{file_prefix}_dec.yuv"
        cmd = self.get_decode_cmd(
            bitstream_path=bitstream_path, yuv_dec_path=yuv_dec_path
        )
        # self.logger.debug(cmd)
        logpath = Path(f"{codec_output_dir}/{file_prefix}_dec.log")

        start = time.time()
        run_cmdline(cmd, logpath=logpath)
        dec_time = time.time() - start
        # self.logger.debug(f"dec_time:{dec_time}")

        self.yuvio.setReader(
            read_path=yuv_dec_path,
            frmWidth=frame_width,
            frmHeight=frame_height,
        )

        nbframes = get_filesize(yuv_dec_path) // (frame_width * frame_height * 2)

        rec_frames = []
        for i in range(nbframes):
            rec_yuv = self.yuvio.read_single_frame(i)
            rec_frames.append(rec_yuv)

        rec_frames = torch.stack(rec_frames)

        minv, maxv = self.min_max_dataset
        rec_frames = min_max_inv_normalization(rec_frames, minv, maxv, bitdepth=10)

        # TODO (fracape) should feature sizes be part of bitstream?
        thisdir = Path(__file__).parent
        fpn_sizes = thisdir.joinpath(
            f"../../data/mpeg-fcvcm/SFU/sfu-fpn-sizes/{self.dataset}.json"
        )
        with fpn_sizes.open("r") as f:
            try:
                json_dict = json.load(f)
            except json.decoder.JSONDecodeError as err:
                print(f'Error reading file "{fpn_sizes}"')
                raise err

        features = self.vision_model.reshape_frame_to_feature_pyramid(
            rec_frames,
            json_dict["fpn"],
            json_dict["subframe_heights"],
            packing_all_in_one=True,
        )

        output = {"data": features}

        return output


@register_codec("x265")
class x265(x264):
    """Encoder / Decoder class for x265 - ffmpeg"""

    def __init__(
        self,
        vision_model: BaseWrapper,
        dataset_name: "str" = "",
        **kwargs,
    ):
        super().__init__(vision_model, dataset_name, **kwargs)

    def get_encode_cmd(
        self,
        inp_yuv_path: Path,
        qp: int,
        bitstream_path: Path,
        width: int,
        height: int,
        nbframes: int = 1,
        frmRate: int = 1,
    ) -> List[Any]:
        cmd = [
            "ffmpeg",
            "-s:v",
            f"{width}x{height}",
            "-i",
            inp_yuv_path,
            "-c:v",
            "hevc",
            "-crf",
            qp,
            "-preset",
            self.preset,
            "-x265-params",
            "bframes=0",
            "-tune",
            self.tune,
            "-pix_fmt",
            "gray10le",
            "-threads",
            "4",
            bitstream_path,
        ]
        return cmd