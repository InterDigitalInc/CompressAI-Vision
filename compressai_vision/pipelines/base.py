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

import errno
import json
import logging
import os
from enum import Enum
from pathlib import Path
from typing import Callable, Dict
from uuid import uuid4 as uuid

import torch
import torch.nn as nn
from torch import Tensor

from compressai_vision.model_wrappers import BaseWrapper


class Parts(Enum):
    def __str__(self):
        return str(self.value)

    NNTaskPart1 = "nn-task-part1"
    Encoder = "encoder"
    Decoder = "decoder"
    NNTaskPart2 = "nn-task-part2"
    Evaluation = "evaluation"


class BasePipeline(nn.Module):
    def __init__(
        self,
        configs: Dict,
        device: str,
    ):
        super().__init__()

        self.logger = logging.getLogger(self.__class__.__name__)
        self.configs = configs
        self.device = device
        self.output_dir = self.configs["output_dir_root"]
        assert self.output_dir, "please provide output directory!"
        self._create_folder(self.output_dir)
        self.bitstream_name = self.configs["codec"]["bitstream_name"]

        self.codec_output_dir = Path(self.configs["codec"]["codec_output_dir"])
        self._create_folder(self.codec_output_dir)
        self.init_time_measure()

    def init_time_measure(self):
        self.elapsed_time = {"nn_part_1": 0, "encode": 0, "decode": 0, "nn_part_2": 0}

    def update_time_elapsed(self, mname, elapsed):
        assert mname in self.elapsed_time
        self.elapsed_time[mname] = self.elapsed_time[mname] + elapsed

    def add_time_details(self, mname: str, details):
        updates = {}
        for k, v in self.elapsed_time.items():
            updates[k] = v
            if k == mname and details is not None:
                assert isinstance(details, dict)
                for sk, sv in details.items():
                    updates[f"{mname}_{sk}"] = sv

        self.elapsed_time = updates

    @property
    def time_elapsed_by_module(self):
        return self.elapsed_time

    @property
    def EXT(self):
        return ".h5"

    @staticmethod
    def _get_title(a):
        return str(a.__class__).split("<class '")[-1].split("'>")[0].split(".")[-1]

    def _update_codec_configs_at_pipeline_level(self, total_num_frames):
        # Sanity check
        self._codec_skip_n_frames = self.configs["codec"]["skip_n_frames"]
        n_frames_to_be_encoded = self.configs["codec"]["n_frames_to_be_encoded"]

        assert (
            self._codec_skip_n_frames < total_num_frames
        ), f"Number of skip frames {self._codec_skip_n_frames} must be less than total number of frames {total_num_frames}"

        if n_frames_to_be_encoded == -1:
            n_frames_to_be_encoded = total_num_frames

        assert (
            n_frames_to_be_encoded
        ), f"Number of frames to be encoded must be greater than 0, but got {n_frames_to_be_encoded}"

        if (self._codec_skip_n_frames + n_frames_to_be_encoded) > total_num_frames:
            self.logger.warning(
                f"The range of frames to be coded is over the total number of frames, {(self._codec_skip_n_frames + n_frames_to_be_encoded)} >= {total_num_frames}"
            )

            n_frames_to_be_encoded = total_num_frames - self._codec_skip_n_frames
            self.logger.warning(
                f"Number of frames to be encoded will be automatically updated. {self._codec_n_frames_to_be_encoded} -> {n_frames_to_be_encoded}"
            )

        self._codec_n_frames_to_be_encoded = n_frames_to_be_encoded

        if (
            self._codec_skip_n_frames > 0
            or self._codec_n_frames_to_be_encoded != total_num_frames
        ):
            assert self.configs["codec"][
                "encode_only"
            ], f"Encoding part of a sequence is only available when `codec.encode_only' is True"

        self._codec_end_frame_idx = (
            self._codec_skip_n_frames + self._codec_n_frames_to_be_encoded
        )

    def _from_input_to_features(
        self, vision_model: BaseWrapper, x: Dict, seq_name: str = None
    ):
        # run NN Part 1 or load pre-computed features
        feature_dir = self.configs["nn_task_part1"].feature_dir

        features_file = f"{feature_dir}/{seq_name}{self.EXT}"

        if (
            self.configs["nn_task_part1"].load_features
            or self.configs["nn_task_part1"].load_features_when_available
        ):
            if Path(features_file).is_file():
                self.logger.debug(f"loading features: {features_file}")
                # features = torch.load(features_file)
                features = torch.load(features_file, map_location=self.device)
            else:
                if self.configs["nn_task_part1"].load_features:
                    raise FileNotFoundError(
                        errno.ENOENT, os.strerror(errno.ENOENT), features_file
                    )
                else:
                    features = vision_model.input_to_features(x)
                    if self.configs["nn_task_part1"].dump_features:
                        self._create_folder(feature_dir)
                        self.logger.debug(f"dumping features in: {feature_dir}")
                        torch.save(features, features_file)
        else:
            features = vision_model.input_to_features(x)
            if self.configs["nn_task_part1"].dump_features:
                self._create_folder(feature_dir)
                self.logger.debug(f"dumping features in: {feature_dir}")
                torch.save(features, features_file)

        return features

    def _from_features_to_output(
        self, vision_model: BaseWrapper, x: Dict, seq_name: str = None
    ):
        """performs the inference of the 2nd part of the NN model"""

        output_results_dir = self.configs["nn_task_part2"].output_results_dir

        results_file = f"{output_results_dir}/{seq_name}{self.EXT}"

        assert "data" in x

        for key, val in x["data"].items():
            if isinstance(val, list):
                if val[0].dim() == 3:
                    x["data"][key] = torch.stack(val)
                else:
                    raise ValueError
            elif isinstance(val, Tensor):  # typical video-pipeline path?
                if val.dim() == 3:
                    x["data"][key] = val.unsqueeze(0)
            else:
                raise ValueError

        if self.configs["conformance"].save_conformance_files:
            self._save_conformance_data(x)

        # suppose that the order of keys and values is matched
        x["data"] = {
            k: v.to(device=self.device)
            for k, v in zip(vision_model.split_layer_list, x["data"].values())
        }

        results = vision_model.features_to_output(x)
        if self.configs["nn_task_part2"].dump_results:
            self._create_folder(output_results_dir)
            torch.save(results, results_file)

        return results

    def _save_conformance_data(self, feature_data: Dict):
        conformance_files_path = self.configs["conformance"].conformance_files_path
        conformance_files_path = self._create_folder(conformance_files_path)
        subsample_ratio = self.configs["conformance"].subsample_ratio

        conformance_data = []
        ch_offset = 0
        for _, data in feature_data["data"].items():
            C = data.shape[1]
            data_means = torch.mean(data, axis=(2, 3)).tolist()[0]
            data_variances = torch.var(data, axis=(2, 3), unbiased=False).tolist()[0]

            subsampled_means = data_means[ch_offset::subsample_ratio]
            subsampled_variances = data_variances[ch_offset::subsample_ratio]
            ch_offset = (ch_offset + C) % subsample_ratio

            conformance_data.append(
                {"means": subsampled_means, "variances": subsampled_variances}
            )

        dump_file_name = Path(feature_data["file_name"]).stem + ".dump"
        json.dump(
            conformance_data,
            open(os.path.join(conformance_files_path, dump_file_name), "w"),
        )

    def _compress(
        self,
        codec,
        x,
        codec_output_dir,
        bitstream_name,
        filename: str,
        remote_inference=False,
    ):
        if self._get_title(codec).lower() == "fctm":
            return codec.encode(
                x,
                codec_output_dir,
                bitstream_name,
                filename,
            )

        return codec.encode(
            x,
            codec_output_dir,
            bitstream_name,
            filename,
            remote_inference=remote_inference,
        )

    def _decompress(
        self,
        codec,
        bitstream,
        codec_output_dir: str,
        filename: str,
        org_img_size: Dict = None,
        remote_inference=False,
    ):
        if self._get_title(codec).lower() == "fctm":
            return codec.decode(
                bitstream,
                codec_output_dir,
                filename,
            )

        return codec.decode(
            bitstream,
            codec_output_dir,
            filename,
            org_img_size=org_img_size,
            remote_inference=remote_inference,
        )

    def _evaluation(self, evaluator: Callable) -> Dict:
        save_path = None
        if self.configs["evaluation"].dump:
            save_path = self._create_folder(self.configs["evaluation"].evaluation_dir)

        if evaluator:
            return evaluator.results(save_path)

        return None

    def _create_folder(self, dir: Path = None) -> Path:
        if dir is None:
            uid = str(uuid())
            path = Path(f"{self.output_dir}/{uid}")
        else:
            path = Path(dir)
        if not path.is_dir():
            self.logger.info(f"creating output folder: {path}")
            path.mkdir(parents=True, exist_ok=True)

        return path

    def _get_model_input_size(self, vision_model: BaseWrapper, x: Dict):
        return vision_model.get_input_size(x)
