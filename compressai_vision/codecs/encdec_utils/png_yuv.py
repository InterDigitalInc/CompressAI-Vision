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

import logging
import math
import shutil
import tempfile

from pathlib import Path
from typing import Dict, Optional

import numpy as np

from PIL import Image

from compressai_vision.utils.external_exec import run_cmdline

from .rawvideo import get_raw_video_file_info


class PngFilesToYuvFileConverter:
    def __init__(
        self,
        chroma_format: str,
        input_bitdepth: int,
        use_yuv: bool,
        frame_rate,
        ffmpeg_loglevel: str,
        logger: logging.Logger,
    ):
        self.chroma_format = chroma_format
        self.input_bitdepth = input_bitdepth
        self.use_yuv = use_yuv
        self.frame_rate = frame_rate
        self.ffmpeg_loglevel = ffmpeg_loglevel
        self.logger = logger

    def __call__(self, input: Dict, file_prefix: str, resize_mapper=None):
        """Converts the input image or video to YUV format using ffmpeg.

        Args:
            input (Dict): A dictionary containing information about the input. It should have the following keys:
                - file_names (List[str]): A list of file names for the input. If it contains more than one file, it is considered a video.
                - last_frame (int): The last frame number of the video.
                - frame_skip (int): The number of frames to skip in the video.
                - org_input_size (Dict[str, int]): A dictionary containing the width and height of the input.
            file_prefix (str): The prefix for the output file name.

        Returns:
            Tuple[str, int, int, int, str]: A tuple containing the following:
                - yuv_in_path (str): The path to the converted YUV input file.
                - nb_frames (int): The number of frames in the input.
                - frame_width (int): The width of the frames in the input.
                - frame_height (int): The height of the frames in the input.
                - file_prefix (str): The updated file prefix.

        Raises:
            AssertionError: If the number of images in the input folder does not match the expected number of frames.
        """
        file_names = input["file_names"]
        temp_dir_obj = None
        resized_info = None
        try:
            if resize_mapper is not None:
                temp_dir_obj = tempfile.TemporaryDirectory(prefix="png2yuv_resized_")
                temp_dir = Path(temp_dir_obj.name)
                resized_file_names = []

                for i, img_file in enumerate(file_names):
                    mapped = resize_mapper({"file_name": img_file})
                    resized_img = mapped["image"]  # expected CHW

                    if hasattr(resized_img, "detach"):
                        resized_img = resized_img.detach().cpu().numpy()

                    if resized_img.ndim != 3:
                        if resized_img.ndim == 4 and resized_img.shape[0] == 1:
                            resized_img = resized_img.squeeze(0)
                        else:
                            raise ValueError(
                                f"Expected CHW image from resize_mapper, got shape={resized_img.shape}"
                            )

                    resized_img = np.transpose(resized_img, (1, 2, 0))  # CHW -> HWC

                    if resized_img.dtype != np.uint8:
                        if np.issubdtype(resized_img.dtype, np.floating):
                            vmin = float(np.nanmin(resized_img))
                            vmax = float(np.nanmax(resized_img))
                            if 0.0 <= vmin and vmax <= 1.0:
                                resized_img = resized_img * 255.0
                        resized_img = np.clip(resized_img, 0, 255).astype(np.uint8)

                    src_ext = Path(img_file).suffix.lower()
                    out_ext = (
                        ".png" if src_ext not in [".png", ".jpg", ".jpeg"] else src_ext
                    )
                    out_file = temp_dir / f"{i:06d}{out_ext}"
                    Image.fromarray(resized_img).save(out_file)
                    resized_file_names.append(str(out_file))

                    if resized_info is None:
                        h, w = resized_img.shape[:2]
                        resized_info = {"width": w, "height": h}

                file_names = resized_file_names

            if len(file_names) > 1:  # video
                # NOTE: using glob for now, should be more robust and look at skipped
                # NOTE: somewhat rigid pattern (lowercase png/jpg)

                parent = Path(file_names[0]).parent
                ext = next(
                    (e for e in ["*.png", "*.jpg", "*.jpeg"] if list(parent.glob(e))),
                    None,
                )
                filename_pattern = f"{parent}/{ext}"
                images_in_folder = len(list(parent.glob(ext)))
                nb_frames = input["last_frame"] - input["frame_skip"]

                assert (
                    images_in_folder == nb_frames
                ), f"input folder contains {images_in_folder} images, {nb_frames} were expected"

                input_info = [
                    "-pattern_type",
                    "glob",
                    "-i",
                    filename_pattern,
                ]

                yuv_file = Path(f"{Path(file_names[0]).parent.parent}.yuv")
                print(f"Checking if YUV is available: {yuv_file}")
                if not yuv_file.is_file():
                    yuv_file = None

            else:
                nb_frames = 1
                input_info = ["-i", file_names[0]]
                yuv_file = None

            chroma_format = self.chroma_format
            input_bitdepth = self.input_bitdepth

            if resized_info is not None:
                src_width = resized_info["width"]
                src_height = resized_info["height"]
            else:
                src_width = input["org_input_size"]["width"]
                src_height = input["org_input_size"]["height"]

            frame_width = math.ceil(src_width / 2) * 2
            frame_height = math.ceil(src_height / 2) * 2
            file_prefix = f"{file_prefix}_{frame_width}x{frame_height}_{self.frame_rate}fps_{input_bitdepth}bit_p{chroma_format}"
            yuv_in_path = f"{file_prefix}_input.yuv"

            chroma_format = "gray" if chroma_format == "400" else f"yuv{chroma_format}p"

            # Use existing YUV (if found and indicated for use):
            if self.use_yuv:
                assert (
                    yuv_file is not None
                ), "Parameter 'use_yuv' set True but YUV file not found."
                size = yuv_file.stat().st_size
                bytes_per_luma_sample = {"yuv420p": 1.5}[chroma_format]
                bytes_per_sample = (input_bitdepth + 7) >> 3
                expected_size = int(
                    frame_width
                    * frame_height
                    * bytes_per_luma_sample
                    * bytes_per_sample
                    * nb_frames
                )
                assert (
                    size == expected_size
                ), f"YUV found for input but expected size of {expected_size} bytes differs from actual size of {size} bytes"
                shutil.copy(yuv_file, yuv_in_path)
                print(f"Using pre-existing YUV file: {yuv_file}")
                return (yuv_in_path, nb_frames, frame_width, frame_height, file_prefix)

            # TODO (fracape)
            # we don't enable skipping frames (codec.skip_n_frames) nor use n_frames_to_be_encoded in video mode
            pix_fmt_suffix = "10le" if input_bitdepth == 10 else ""

            convert_cmd = [
                "ffmpeg",
                "-y",
                "-hide_banner",
                "-loglevel",
                f"{self.ffmpeg_loglevel}",
            ]
            convert_cmd += input_info
            convert_cmd += [
                "-vf",
                "pad=ceil(iw/2)*2:ceil(ih/2)*2",
                "-f",
                "rawvideo",
                "-pix_fmt",
                f"{chroma_format}{pix_fmt_suffix}",
                "-dst_range",
                "1",  #  (fracape) convert to full range for now
            ]

            convert_cmd.append(yuv_in_path)
            self.logger.debug(convert_cmd)

            run_cmdline(convert_cmd)

            return yuv_in_path, nb_frames, frame_width, frame_height, file_prefix
        finally:
            if temp_dir_obj is not None:
                temp_dir_obj.cleanup()


class YuvFileToPngFilesConverter:
    def __init__(self, datacatalog: str, logger: logging.Logger):
        self.datacatalog = datacatalog
        self.logger = logger

    def __call__(
        self,
        output_file_prefix: str,
        dec_path: str,
        yuv_dec_path: Path,
        org_img_size: Optional[Dict] = None,
        vcm_mode: bool = False,
    ):
        """Converts a YUV file to a series of PNG images using ffmpeg.

        Args:
            output_file_prefix (str): The prefix of the output file name.
            dec_path (str): The path to the directory where the PNG images will be saved.
            yuv_dec_path (Path): The path to the input YUV file.
            org_img_size (Dict, optional): The original image size. Defaults to None.

        Returns:
            None

        Raises:
            AssertionError: If the video format is not YUV420.
        """
        video_info = get_raw_video_file_info(output_file_prefix.split("qp")[-1])
        frame_width = video_info["width"]
        frame_height = video_info["height"]

        assert (
            "420" in video_info["format"].value
        ), f"Only support yuv420, but got {video_info['format']}"
        pix_fmt_suffix = "10le" if video_info["bitdepth"] == 10 else ""
        chroma_format = "yuv420p"

        cmd_suffix, output_png_filename = self._determine_output_filename(
            output_file_prefix
        )

        output_png = f"{dec_path}/{output_png_filename}"

        convert_cmd = [
            "ffmpeg",
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-f",
            "rawvideo",
            "-pix_fmt",
            f"{chroma_format}{pix_fmt_suffix}",
            "-s",
            f"{frame_width}x{frame_height}",
        ]

        if not vcm_mode:
            convert_cmd.extend(
                [
                    "-src_range",
                    "1",  # (fracape) assume dec yuv is full range for now
                ]
            )

        convert_cmd.extend(
            [
                "-i",
                f"{yuv_dec_path}",
                "-pix_fmt",
                "rgb24",
            ]
        )

        if vcm_mode:
            convert_cmd.extend(
                [
                    "-vsync",
                    "1",
                ]
            )
        convert_cmd.extend(
            [
                *cmd_suffix,
                output_png,
            ]
        )

        run_cmdline(convert_cmd)

        if org_img_size is not None and (
            frame_height != org_img_size["height"]
            or frame_width != org_img_size["width"]
        ):
            self.logger.warning(
                f"Different original input size found. "
                f"It must be {org_img_size['width']}x{org_img_size['height']}, "
                f"but {frame_width}x{frame_height} are parsed from YUV"
            )

            self.logger.warning(
                f"Use {org_img_size['width']}x{org_img_size['height']}, "
                f"instead of {frame_width}x{frame_height}"
            )

            self._crop_decoded_png(output_png, dec_path, org_img_size)

    def _crop_decoded_png(self, output_png: str, dec_path: str, org_img_size: Dict):
        tmp_output_png = f"{dec_path}/{Path(output_png).stem}_tmp.png"

        convert_cmd = [
            "ffmpeg",
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            output_png,
            "-vf",
            f"crop={org_img_size['width']}:{org_img_size['height']}",
            tmp_output_png,  # no name change
        ]
        run_cmdline(convert_cmd)

        Path(output_png).unlink()
        Path(tmp_output_png).rename(output_png)

    def _determine_output_filename(self, output_file_prefix: str):
        datacatalog = self.datacatalog

        # TODO (fracape) hacky, clean this
        if datacatalog in ["MPEGOIV6", "MPEGSAM"]:
            cmd_suffix = []
            filename = f"{output_file_prefix}.png"
        elif datacatalog == "SFUHW":
            cmd_suffix = ["-start_number", "0"]
            prefix = output_file_prefix.split("qp")[0]
            filename = f"{prefix}%03d.png"
        elif datacatalog in ["MPEGHIEVE", "PANDASET"]:
            cmd_suffix = ["-start_number", "0"]
            filename = "%06d.png"
        elif datacatalog in ["MPEGTVDTRACKING"]:
            cmd_suffix = ["-start_number", "1"]
            filename = "%06d.png"
        else:
            raise ValueError(f"Unknown datacatalog: {datacatalog}")

        return cmd_suffix, filename
