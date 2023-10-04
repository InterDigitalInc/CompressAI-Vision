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

import logging
import os
import shutil
import time
from enum import Enum
from pathlib import Path
from typing import Callable, Dict
from uuid import uuid4 as uuid

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from compressai_vision.evaluators import BaseEvaluator
from compressai_vision.model_wrappers import BaseWrapper
from compressai_vision.registry import register_pipeline
from compressai_vision.utils import dataio

from ..base import BasePipeline

""" A schematic for the split-inference pipline

.. code-block:: none

     ┌─────────────────┐                                         ┌─────────────────┐
     │                 │     ┌───────────┐     ┌───────────┐     │                 │
     │     NN Task     │     │           │     │           │     │      NN Task    │
────►│                 ├────►│  Encoder  ├────►│  Decoder  ├────►│                 ├────►
     │      Part 1     │     │           │     │           │     │      Part 2     │
     │                 │     └───────────┘     └───────────┘     │                 │
     └─────────────────┘                                         └─────────────────┘

    ──────►──────►──────►──────►──────►──────►──────►──────►──────►──────►
"""


@register_pipeline("image-split-inference")
class ImageSplitInference(BasePipeline):
    def __init__(
        self,
        configs: Dict,
        device: str,
    ):
        super().__init__(configs, device)

    def __call__(
        self,
        vision_model: BaseWrapper,
        codec,
        dataloader: DataLoader,
        evaluator: BaseEvaluator,
    ) -> Dict:
        """Push image(s) through the encoder+decoder, returns number of bits for each image and encoded+decoded images

        Returns (nbitslist, x_hat), where nbitslist is a list of number of bits and x_hat is the image that has gone throught the encoder/decoder process
        """
        self._update_codec_configs_at_pipeline_level(len(dataloader))
        output_list = []
        timing = {"nn_part_1": 0, "encode": 0, "decode": 0, "nn_part_2": 0}
        for e, d in enumerate(tqdm(dataloader)):
            org_img_size = {"height": d[0]["height"], "width": d[0]["width"]}
            file_prefix = f'img_id_{d[0]["image_id"]}'

            if not self.configs["codec"]["decode_only"]:
                if e < self._codec_skip_n_frames:
                    continue
                if e >= self._codec_end_frame_idx:
                    break

                start = time.time()
                featureT = self._from_input_to_features(vision_model, d, file_prefix)
                end = time.time()
                timing["nn_part_1"] = timing["nn_part_1"] + (end - start)

                featureT["org_input_size"] = org_img_size

                start = time.time()
                res = self._compress_features(
                    codec,
                    featureT,
                    self.codec_output_dir,
                    self.bitstream_name,
                    file_prefix,
                )
                end = time.time()
                timing["encode"] = timing["encode"] + (end - start)
            else:
                res = {}
                bitstream_name = f"{self.bitstream_name}-{file_prefix}.bin"
                bitstream_path = os.path.join(self.codec_output_dir, bitstream_name)
                print(f"reading bitstream... {bitstream_path}")
                res["bitstream"] = bitstream_path

            if self.configs["codec"]["encode_only"] is True:
                continue

            start = time.time()
            dec_features = self._decompress_features(
                codec, res["bitstream"], self.codec_output_dir, file_prefix
            )
            end = time.time()
            timing["decode"] = timing["decode"] + (end - start)

            # dec_features should contain "org_input_size" and "input_size"
            # When using anchor codecs, that's not the case, we read input images to derive them
            if not "org_input_size" in dec_features or not "input_size" in dec_features:
                self.logger.warning(
                    "Hacky: 'org_input_size' and 'input_size' retrived from input dataset."
                )
                dec_features["org_input_size"] = org_img_size
                dec_features["input_size"] = self._get_model_input_size(vision_model, d)

            dec_features["file_name"] = d[0]["file_name"]

            start = time.time()
            pred = self._from_features_to_output(
                vision_model, dec_features, file_prefix
            )
            end = time.time()
            timing["nn_part_2"] = timing["nn_part_2"] + (end - start)

            evaluator.digest(d, pred)

            out_res = d[0].copy()
            del (
                out_res["image"],
                out_res["width"],
                out_res["height"],
                out_res["image_id"],
            )
            out_res["qp"] = (
                "uncmp" if codec.qp_value is None else codec.qp_value
            )  # Assuming one qp will be used
            if self.configs["codec"]["decode_only"]:
                out_res["bytes"] = os.stat(res["bitstream"]).st_size
            else:
                out_res["bytes"] = res["bytes"][0]
            out_res["coded_order"] = e
            out_res["org_input_size"] = f'{d[0]["height"]}x{d[0]["width"]}'
            out_res["input_size"] = dec_features["input_size"][0]
            output_list.append(out_res)

        if self.configs["codec"]["encode_only"] is True:
            print(f"bitstreams generated, exiting")
            raise SystemExit(0)

        eval_performance = self._evaluation(evaluator)

        return timing, codec.eval_encode_type, output_list, eval_performance
