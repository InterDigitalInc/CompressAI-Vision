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
from typing import Callable, Dict, Tuple
from uuid import uuid4 as uuid

import torch
import torch.nn as nn
from omegaconf.errors import InterpolationResolutionError
from torch import Tensor

from compressai_vision.codecs.utils import (
    MIN_MAX_DATASET,
    min_max_inv_normalization,
    min_max_normalization,
)
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
        device: Dict,
    ):
        super().__init__()

        self.logger = logging.getLogger(self.__class__.__name__)
        self.configs = configs

        assert isinstance(device, Dict)

        self.device_nn_part1 = device["nn_part1"]
        self.device_nn_part2 = device["nn_part2"]

        self.output_dir = self.configs["output_dir_root"]
        assert self.output_dir, "please provide output directory!"
        self._create_folder(self.output_dir)
        self.bitstream_name = self.configs["codec"]["bitstream_name"]
        self._output_ext = ".h5"

        try:
            vis_flag = self.configs["visualization"].save_visualization
        except InterpolationResolutionError:
            vis_flag = False
        if vis_flag:
            self.vis_dir = self.configs["visualization"].visualization_dir
            self.vis_threshold = self.configs["visualization"].get("threshold", None)
            self._create_folder(self.vis_dir)

        self.codec_output_dir = Path(self.configs["codec"]["codec_output_dir"])
        self.is_mac_calculation = self.configs["codec"]["measure_complexity"]
        self._create_folder(self.codec_output_dir)
        self.init_time_measure()
        self.init_complexity_measure()
        self.eval()

    def init_time_measure(self):
        self.elapsed_time = {"nn_part_1": 0, "encode": 0, "decode": 0, "nn_part_2": 0}

    def update_time_elapsed(self, mname, elapsed):
        assert mname in self.elapsed_time
        self.elapsed_time[mname] = self.elapsed_time[mname] + elapsed

    def init_complexity_measure(self):
        self.kmacs = {
            "nn_part_1": 0,
            "feature_reduction": 0,
            "feature_restoration": 0,
            "nn_part_2": 0,
        }
        self.pixels = {
            "nn_part_1": 0,
            "feature_reduction": 0,
            "feature_restoration": 0,
            "nn_part_2": 0,
        }

    def add_kmac_and_pixels_info(self, mname, kmac, pixels):
        assert mname in self.kmacs
        self.kmacs[mname] = kmac
        self.pixels[mname] = pixels

    def acc_kmac_and_pixels_info(self, mname, kmac, pixels):  # for image task
        # accumulate
        assert mname in self.kmacs
        self.kmacs[mname] = self.kmacs[mname] + kmac
        self.pixels[mname] = self.pixels[mname] + pixels

    def calc_kmac_per_pixels_image_task(self):  # for video task
        # multiplication
        self.kmac_per_pixels = {
            k: (v / self.pixels["nn_part_1"]) for k, v in self.kmacs.items()
        }

    def calc_kmac_per_pixels_video_task(self, nbframes, ori_nbframes):  # for video task
        # multiplication
        self.kmac_per_pixels = {
            k: (v * nbframes) / (self.pixels["nn_part_1"] * ori_nbframes)
            for k, v in self.kmacs.items()
        }

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
    def complexity_calc_by_module(self):
        return (
            self.kmac_per_pixels if hasattr(self, "kmac_per_pixels") is True else None
        )

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
            ], "Encoding part of a sequence is only available when `codec.encode_only' is True"

        self._codec_end_frame_idx = (
            self._codec_skip_n_frames + self._codec_n_frames_to_be_encoded
        )

    @staticmethod
    def _prep_features_to_dump(features, n_bits, datacatalog_name):
        output_features = features.copy()
        assert "data" in output_features
        del output_features["data"]

        if n_bits == -1:
            data_features = features["data"]
        elif n_bits >= 8:
            assert (
                n_bits == 8 or n_bits == 16
            ), "currently it only supports dumping features in 8 bits or 16 bits"
            assert datacatalog_name in list(
                MIN_MAX_DATASET.keys()
            ), f"{datacatalog_name} does not exist in the pre-computed minimum and maximum tables"
            minv, maxv = MIN_MAX_DATASET[datacatalog_name]
            data_features = {}
            for key, data in features["data"].items():
                assert (
                    data.min() >= minv and data.max() <= maxv
                ), f"{data.min()} should be greater than {minv} and {data.max()} should be less than {maxv}"
                out, _ = min_max_normalization(data, minv, maxv, bitdepth=n_bits)

                if n_bits <= 8:
                    data_features[key] = out.to(torch.uint8)
                elif n_bits <= 16:
                    data_features[key] = {
                        "lsb": torch.bitwise_and(
                            out.to(torch.int32), torch.tensor(0xFF)
                        ).to(torch.uint8),
                        "msb": torch.bitwise_and(
                            torch.bitwise_right_shift(out.to(torch.int32), 8),
                            torch.tensor(0xFF),
                        ).to(torch.uint8),
                    }
                else:
                    raise NotImplementedError
        else:
            raise NotImplementedError

        output_features["data"] = data_features
        return output_features

    @staticmethod
    def _post_process_loaded_features(features, n_bits, datacatalog_name):
        if n_bits == -1:
            assert "data" in features
        elif n_bits >= 8:
            assert (
                n_bits == 8 or n_bits == 16
            ), "currently it only supports dumping features in 8 bits or 16 bits"
            assert datacatalog_name in list(
                MIN_MAX_DATASET.keys()
            ), f"{datacatalog_name} does not exist in the pre-computed minimum and maximum tables"
            minv, maxv = MIN_MAX_DATASET[datacatalog_name]
            data_features = {}
            for key, data in features["data"].items():
                if n_bits <= 8:
                    out = min_max_inv_normalization(data, minv, maxv, bitdepth=n_bits)
                    data_features[key] = out.to(torch.float32)
                elif n_bits <= 16:
                    lsb_part = data["lsb"].to(torch.int32)
                    msb_part = torch.bitwise_left_shift(data["msb"].to(torch.int32), 8)
                    recovery = (msb_part + lsb_part).to(torch.float32)

                    out = min_max_inv_normalization(
                        recovery, minv, maxv, bitdepth=n_bits
                    )
                    data_features[key] = out.to(torch.float32)
                else:
                    raise NotImplementedError

            features["data"] = data_features
        else:
            raise NotImplementedError

        return features

    def _from_input_to_features(
        self,
        vision_model: BaseWrapper,
        x: Dict,
        seq_name: str = None,
        datacatalog_name=None,
    ):
        # run NN Part 1 or load pre-computed features
        feature_dir = self.configs["nn_task_part1"].feature_dir

        features_file = f"{feature_dir}/{seq_name}{self._output_ext}"

        if (
            self.configs["nn_task_part1"].load_features
            or self.configs["nn_task_part1"].load_features_when_available
        ):
            if Path(features_file).is_file():
                self.logger.debug(f"loading features: {features_file}")
                # features = torch.load(features_file)
                features = torch.load(features_file, map_location=self.device_nn_part1)
                features = self._post_process_loaded_features(
                    features,
                    self.configs["nn_task_part1"].load_features_n_bits,
                    datacatalog_name,
                )
            else:
                if self.configs["nn_task_part1"].load_features:
                    raise FileNotFoundError(
                        errno.ENOENT, os.strerror(errno.ENOENT), features_file
                    )
                else:
                    features = vision_model.input_to_features(x, self.device_nn_part1)
                    if self.configs["nn_task_part1"].dump_features:
                        self._create_folder(feature_dir)
                        self.logger.debug(f"dumping features in: {feature_dir}")
                        features_to_dump = self._prep_features_to_dump(
                            features,
                            self.configs["nn_task_part1"].dump_features_n_bits,
                            datacatalog_name,
                        )
                        torch.save(features_to_dump, features_file)
        else:
            features = vision_model.input_to_features(x, self.device_nn_part1)
            if self.configs["nn_task_part1"].dump_features:
                self._create_folder(feature_dir)
                self.logger.debug(f"dumping features in: {feature_dir}")
                features_to_dump = self._prep_features_to_dump(
                    features,
                    self.configs["nn_task_part1"].dump_features_n_bits,
                    datacatalog_name,
                )
                torch.save(features_to_dump, features_file)

        return features

    def _from_features_to_output(
        self, vision_model: BaseWrapper, x: Dict, seq_name: str = None
    ):
        """performs the inference of the 2nd part of the NN model"""
        output_results_dir = self.configs["nn_task_part2"].output_results_dir

        results_file = f"{output_results_dir}/{seq_name}{self._output_ext}"

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
            k: v.to(device=self.device_nn_part2)
            for k, v in zip(vision_model.split_layer_list, x["data"].values())
        }

        results = vision_model.features_to_output(x, self.device_nn_part2)
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
        vcm_mode=False,
        output10b=False,
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
            vcm_mode=vcm_mode,
            output10b=output10b,
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

    def calc_feature_mse(
        self,
        input_feats: Dict[str, torch.Tensor],
        recon_feats: Dict[str, torch.Tensor],
    ) -> Dict[str, float]:

        mse_results: Dict[str, float] = {}

        keys_recon = list(recon_feats.keys())

        for i, key in enumerate(input_feats.keys()):

            x = input_feats[key].cpu()
            y = recon_feats[keys_recon[i]].cpu()

            assert (
                x.shape == y.shape
            ), f"Shape mismatch at {key}: {x.shape} vs {y.shape}"

            mse = torch.mean((x - y) ** 2).item()
            mse_results[key] = mse

        return mse_results

    def _get_prompts(self, vision_model: BaseWrapper, x: Dict):
        return vision_model.get_prompts(x)

    def _get_object_classes(self, vision_model: BaseWrapper, x: Dict):
        return vision_model.get_object_classes(x)
