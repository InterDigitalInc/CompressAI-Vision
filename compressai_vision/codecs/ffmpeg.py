# Copyright (c) 2022-2024, InterDigital Communications, Inc
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

import configparser
import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Union

import torch
import torch.nn as nn

from compressai_vision.codecs.utils import FpnUtils
from compressai_vision.model_wrappers import BaseWrapper
from compressai_vision.registry import register_codec
from compressai_vision.utils import time_measure
from compressai_vision.utils.dataio import PixelFormat, readwriteYUV
from compressai_vision.utils.external_exec import run_cmdline

from .encdec_utils import get_raw_video_file_info
from .utils import MIN_MAX_DATASET, min_max_inv_normalization, min_max_normalization


def get_filesize(filepath: Union[Path, str]) -> int:
    return Path(filepath).stat().st_size


@register_codec("x264")
class x264(nn.Module):
    """Encoder/Decoder class for x265"""

    def __init__(
        self,
        vision_model: BaseWrapper,
        dataset: Dict,
        **kwargs,
    ):
        super().__init__()

        self.qp = kwargs["encoder_config"]["qp"]

        self.eval_encode = kwargs["eval_encode"]

        self.dump = kwargs["dump"]
        self.vision_model = vision_model
        self.datacatalog = dataset.datacatalog
        self.dataset_name = dataset.config["dataset_name"]

        self.frame_rate = 1
        if not self.datacatalog == "MPEGOIV6":
            config = configparser.ConfigParser()
            config.read(f"{dataset['config']['root']}/{dataset['config']['seqinfo']}")
            self.frame_rate = config["Sequence"]["frameRate"]

        if self.datacatalog in MIN_MAX_DATASET:
            self.min_max_dataset = MIN_MAX_DATASET[self.datacatalog]
        elif self.dataset_name in MIN_MAX_DATASET:
            self.min_max_dataset = MIN_MAX_DATASET[self.dataset_name]
        else:
            raise ValueError("dataset not recognized for normalization")

        # TODO (fracape) bitdepth in cfg
        self.colorformat = "444"
        self.yuvio = readwriteYUV(device="cpu", format=PixelFormat.YUV444_10le)

        self.preset = kwargs["encoder_config"]["preset"]
        self.tune = kwargs["encoder_config"]["tune"]

        self.logger = logging.getLogger(self.__class__.__name__)
        self.verbosity = kwargs["verbosity"]
        logging_level = logging.WARN
        if self.verbosity == 1:
            logging_level = logging.INFO
        if self.verbosity >= 2:
            logging_level = logging.DEBUG

        self.logger.setLevel(logging_level)

        self.fpn_utils = FpnUtils()

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
        frmRate: int = 1,
    ) -> List[Any]:
        """
        Generates the ffmpeg command (x264 lib) for encoding the input YUV video to H.264 format with the specified parameters.
        Args:
            inp_yuv_path (Path): The input YUV video file path.
            qp (int): The quantization parameter for the video encoding.
            bitstream_path (Path): The output bitstream file path.
            width (int): The width of the video frame.
            height (int): The height of the video frame.
            frmRate (int, optional): The frame rate of the video. Defaults to 1.
        Returns:
            List[Any]: The generated ffmpeg command for encoding the video.
        """
        cmd = [
            "ffmpeg",
            "-y",
            "-s:v",
            f"{width}x{height}",
            "-framerate",
            f"{frmRate}",
            "-i",
            f"{inp_yuv_path}",
            "-c:v",
            "h264",
            "-crf",
            f"{qp}",
            "-preset",
            self.preset,
            "-bf",
            "0",
            "-tune",
            self.tune,
            "-pix_fmt",
            "yuv444p10le",  # to be checked
            "-threads",
            "4",
            f"{bitstream_path}",
        ]
        return cmd

    def get_decode_cmd(self, bitstream_path: Path, yuv_dec_path: Path) -> List[Any]:
        """
        Get the ffmpeg decode command (x264 lib) for the given bitstream path and YUV decode path.
        Args:
            bitstream_path (Path): The path to the bitstream file.
            yuv_dec_path (Path): The path to save the YUV decoded file.
        Returns:
            List[Any]: The list containing the ffmpeg command for decoding.
        """
        cmd = [
            "ffmpeg",
            "-y",
            "-i",
            f"{bitstream_path}",
            "-pix_fmt",
            "yuv444p10le",
            f"{yuv_dec_path}",
        ]
        return cmd

    def encode(
        self,
        x: Dict,
        codec_output_dir,
        bitstream_name,
        file_prefix: str = "",
        remote_inference=False,
    ) -> bool:
        """
        Encodes the input feature tensors and returns the encoded bitstream.
        Args:
            x (Dict): The input data dictionary.
            codec_output_dir (str): The directory for codec output.
            bitstream_name (str): The name of the bitstream.
            file_prefix (str, optional): The prefix for the file. Defaults to "".
            remote_inference (bool): Flag for remote inference.
        Returns:
            Dict: numbers of bytes per frame and bitstream path.
        """
        assert not remote_inference  # TODO (fracape) remote inference not supported yet
        bitdepth = 10  # TODO (fracape) (add this as config)

        start = time.time()

        frames = self.fpn_utils.reshape_feature_pyramid_to_frame(
            x["data"], packing_all_in_one=True
        )
        minv, maxv = self.min_max_dataset
        frames, mid_level = min_max_normalization(frames, minv, maxv, bitdepth=bitdepth)

        conversion_time = time.time() - start
        self.logger.debug(f"conversion time:{conversion_time}")

        nbframes, frame_height, frame_width = frames.size()
        frmRate = self.frame_rate if nbframes > 1 else 1

        if file_prefix == "":
            file_prefix = f"{codec_output_dir}/{bitstream_name}"
        else:
            file_prefix = f"{codec_output_dir}/{bitstream_name}-{file_prefix}"

        file_prefix = f"{file_prefix}_{frame_width}x{frame_height}_{frmRate}fps_{bitdepth}bit_p{self.colorformat}"

        yuv_in_path = f"{file_prefix}_input.yuv"
        # yuv_in_converted_path = f"{file_prefix}_input_420.yuv"
        bitstream_path = f"{file_prefix}.mp4"
        logpath = Path(f"{file_prefix}_enc.log")
        # convert_logpath = Path(f"{file_prefix}_convert.log")

        self.yuvio.setWriter(
            write_path=yuv_in_path,
            frmWidth=frame_width,
            frmHeight=frame_height,
        )
        for i, frame in enumerate(frames):
            self.yuvio.write_one_frame(frame, mid_level=mid_level, frame_idx=i)

        cmd = self.get_encode_cmd(
            yuv_in_path,
            width=frame_width,
            height=frame_height,
            qp=self.qp,
            bitstream_path=bitstream_path,
            frmRate=frmRate,
        )
        # TOTO logger
        # self.logger.debug(cmd)

        start = time.time()
        run_cmdline(cmd, logpath=logpath)
        enc_time = time.time() - start
        # self.logger.debug(f"enc_time:{enc_time}")

        if not self.dump["dump_yuv_input"]:
            Path(yuv_in_path).unlink()
            # Path(yuv_in_converted_path).unlink()

        # to be compatible with the pipelines
        # per frame bits can be collected by parsing enc log to be more accurate
        avg_bytes_per_frame = get_filesize(bitstream_path) / nbframes
        all_bytes_per_frame = [avg_bytes_per_frame] * nbframes

        output = {
            "bytes": all_bytes_per_frame,
            "bitstream": bitstream_path,
        }
        enc_times = {
            "video": enc_time,
            "conversion": conversion_time,
        }

        mac_calculations = None  # no NN-related complexity calculation with std codecs

        return output, enc_times, mac_calculations

    def decode(
        self,
        bitstream_path: Path = None,
        codec_output_dir: str = "",
        file_prefix: str = "",
        org_img_size: Dict = None,
        remote_inference=False,
        vcm_mode=False,
    ) -> bool:
        """
        Decodes a bitstream into video frames and extract features from the decoded frames.
        Args:
            bitstream_path (Path): The path to the input bitstream file.
            codec_output_dir (str): The directory where the codec output will be stored.
            file_prefix (str): The prefix to be used for the output files.

        Returns:
            Dict: dictionary of output features.
        """
        assert not remote_inference  # TODO (fracape) remote inference not supported yet
        assert not vcm_mode  # TODO To be developed upon supporting remote inference.
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

        # warning expect 10bit, i.e. uint16 444
        nbframes = int(
            get_filesize(yuv_dec_path) // (frame_width * frame_height * 2 * 3)
        )

        rec_frames = []
        for i in range(nbframes):
            rec_yuv = self.yuvio.read_one_frame(i)
            rec_frames.append(rec_yuv)

        rec_frames = torch.stack(rec_frames)

        start = time_measure()
        minv, maxv = self.min_max_dataset
        rec_frames = min_max_inv_normalization(rec_frames, minv, maxv, bitdepth=10)

        # TODO (fracape) should feature sizes be part of bitstream even for anchors
        thisdir = Path(__file__).parent
        if self.datacatalog == "MPEGOIV6":
            fpn_sizes = thisdir.joinpath(
                f"../../data/mpeg-fcm/{self.datacatalog}/fpn-sizes/{self.dataset_name}/{file_prefix}.json"
            )
        else:
            fpn_sizes = thisdir.joinpath(
                f"../../data/mpeg-fcm/{self.datacatalog}/fpn-sizes/{self.dataset_name}.json"
            )
        with fpn_sizes.open("r") as f:
            try:
                json_dict = json.load(f)
            except json.decoder.JSONDecodeError as err:
                print(f'Error reading file "{fpn_sizes}"')
                raise err

        features = self.fpn_utils.reshape_frame_to_feature_pyramid(
            rec_frames,
            json_dict["fpn"],
            json_dict["subframe_heights"],
            packing_all_in_one=True,
        )

        conversion_time = time_measure() - start
        self.logger.debug(f"conversion_time:{conversion_time}")

        if not self.dump["dump_yuv_packing_dec"]:
            Path(yuv_dec_path).unlink()

        output = {"data": features}
        dec_times = {
            "video": dec_time,
            "conversion": conversion_time,
        }

        mac_calculations = None  # no NN-related complexity calculation with std codecs

        return output, dec_times, mac_calculations


@register_codec("x265")
class x265(x264):
    """Encoder / Decoder class for x265 - ffmpeg"""

    def __init__(
        self,
        vision_model: BaseWrapper,
        dataset: Dict,
        **kwargs,
    ):
        super().__init__(vision_model, dataset, **kwargs)
        self.colorformat = "444"
        self.yuvio = readwriteYUV(device="cpu", format=PixelFormat.YUV444_10le)

    def get_encode_cmd(
        self,
        inp_yuv_path: Path,
        qp: int,
        bitstream_path: Path,
        width: int,
        height: int,
        frmRate: int = 1,
    ) -> List[Any]:
        """
        Generates the ffmpeg command (x265 lib) for encoding the input YUV video to H.265 format with the specified parameters.
        Args:
            inp_yuv_path (Path): The input YUV video file path.
            qp (int): The quantization parameter for the video encoding.
            bitstream_path (Path): The output bitstream file path.
            width (int): The width of the video frame.
            height (int): The height of the video frame.
            frmRate (int, optional): The frame rate of the video. Defaults to 1.
        Returns:
            List[Any]: The generated ffmpeg command for encoding the video.
        """
        cmd = [
            "ffmpeg",
            "-y",
            "-s:v",
            f"{width}x{height}",
            "-framerate",
            f"{frmRate}",
            "-i",
            f"{inp_yuv_path}",
            "-c:v",
            "hevc",
            "-crf",
            f"{qp}",
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
            f"{bitstream_path}",
        ]
        return cmd
