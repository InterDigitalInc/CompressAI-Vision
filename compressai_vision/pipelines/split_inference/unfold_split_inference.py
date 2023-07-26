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

from .base_split import BaseSplit


@register_pipeline("unfold-split-inference")
class UnfoldSplitInference(BaseSplit):
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
        output_list = []
        for e, d in enumerate(tqdm(dataloader)):
            # TODO [hyomin - Make DefaultDatasetLoader compatible with Detectron2DataLoader]
            # Please reference to Detectron2 Dataset Mapper. Will face an issue when supporting Non-Detectron2-based network such as YOLO.

            org_img_size = {"height": d[0]["height"], "width": d[0]["width"]}

            file_prefix = f'img_id_{d[0]["image_id"]}'

            featureT = self._from_input_to_features(vision_model, d, file_prefix)
            featureT["org_input_size"] = org_img_size

            res = self._compress_features(codec, featureT, file_prefix)

            dec_featureT = self._decompress_features(
                codec, res["bitstream"], file_prefix
            )

            # TODO (hyomin) how should org_input_size and input_size be conveyed, bitstream?
            if not "data" in dec_featureT:
                dec_featureT = {
                    "data": dec_featureT,
                    "input_size": featureT["input_size"],
                    "org_input_size": featureT["org_input_size"],
                }

            dec_featureT["file_name"] = d[0]["file_name"]
            pred = self._from_features_to_output(
                vision_model, dec_featureT, file_prefix
            )

            evaluator.digest(d, pred)

            out_res = d[0].copy()
            del out_res["image"], out_res["width"], out_res["height"]
            out_res["bytes"] = res["bytes"][0]
            out_res["coded_order"] = e
            out_res["org_input_size"] = (d[0]["height"], d[0]["width"])
            out_res["input_size"] = featureT["input_size"][0]
            output_list.append(out_res)

        mAP = self._evaluation(evaluator)

        return {"coded_res": output_list, "mAP": mAP}
