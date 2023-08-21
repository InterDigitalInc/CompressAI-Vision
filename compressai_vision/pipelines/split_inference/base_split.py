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
import shutil
from enum import Enum
from pathlib import Path
from typing import Callable, Dict
from uuid import uuid4 as uuid

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

# from compressai_vision.evaluators import BaseEvaluator
from compressai_vision.model_wrappers import BaseWrapper

# from compressai_vision.registry import register_pipeline
# from compressai_vision.utils import dataio

EXT = ".h5"


class Parts(Enum):
    def __str__(self):
        return str(self.value)

    NNTaskPart1 = "nn-task-part1"
    Encoder = "encoder"
    Decoder = "decoder"
    NNTaskPart2 = "nn-task-part2"
    Evaluation = "evaluation"


""" A schematic for the split-inference pipline

.. code-block:: none

             Fold                                                        Fold
           ┌ ─── ┐                                                     ┌ ─── ┐
           |     |                                                     |     |
           |     │                                                     |     │
     ┌─────┴─────▼─────┐                                         ┌─────┴─────▼─────┐
     │                 │     ┌───────────┐     ┌───────────┐     │                 │
     │     NN Task     │     │           │     │           │     │      NN Task    │
────►│                 ├────►│  Encoder  ├────►│  Decoder  ├────►│                 ├────►
     │      Part 1     │     │           │     │           │     │      Part 2     │
     │                 │     └───────────┘     └───────────┘     │                 │
     └─────────────────┘                                         └─────────────────┘

    ──────►──────►──────►──────►──────►───Unfold───►──────►──────►──────►──────►──────►
"""


class BaseSplit(nn.Module):
    def __init__(
        self,
        configs: Dict,
        device: str,
    ):
        super().__init__()

        self.logger = logging.getLogger(self.__class__.__name__)
        self.configs = configs
        self.device = device
        self.output_dir = self.configs["output_dir"]
        assert self.output_dir, "please provide output directory!"
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)

    def _from_input_to_features(
        self, vision_model: BaseWrapper, x: Dict, seq_name: str = None
    ):
        # run NN Part 1 or load pre-computed features
        feature_dir = self.configs["nn_task_part1"].feature_dir

        features_file = f"{feature_dir}/{seq_name}{EXT}"

        if self.configs["nn_task_part1"].load_features:
            if not Path(features_file).is_file():
                raise FileNotFoundError(
                    errno.ENOENT, os.strerror(errno.ENOENT), features_file
                )
            self.logger.debug(f"loading features: {features_file}")
            features = torch.load(features_file)
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

        results_file = f"{output_results_dir}/{seq_name}{EXT}"

        assert "data" in x
        if self.configs["conformance"].save_conformance_files:
            self._save_conformance_data(x)

        # for _, tensor in x["data"].items():
        #     tensor.to(self.device)
        x["data"] = {k: v.to(device=self.device) for k, v in x["data"].items()}

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
            N, C, H, W = data.shape
            data_means = torch.mean(data, axis=(2, 3)).tolist()[0]
            data_variances = torch.var(data, axis=(2, 3)).tolist()[0]

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

    def _compress_features(self, codec, x, filename: str):
        return codec.encode(
            x,
            filename,
        )

    def _decompress_features(self, codec, bitstream, filename: str):
        return codec.decode(
            bitstream,
            filename,
        )

    def _evaluation(self, evaluator: Callable) -> Dict:
        save_path = None
        if self.configs["evaluation"].dump:
            if self.configs["evaluation"].evaluation_dir is None:
                save_path = self._create_folder(self.configs["evaluation"])

        out = evaluator.results(save_path)

        return out

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


# TODO (fracape, choih) this might be needed later
# def __del__(self):
#     if self.dump_features:
#         return
#     if not self.chache_features:
#         return

#     self.logger.debug("removing %s", self._caching_folder)
#     shutil.rmtree(self._caching_folder)
