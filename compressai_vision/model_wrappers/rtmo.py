# Copyright (c) 2025, InterDigital Communications, Inc
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
from enum import Enum
from pathlib import Path
from typing import Dict, List

import torch
from mmengine.config import Config
from mmengine.registry import MODELS, DefaultScope

from compressai_vision.registry import register_vision_model

from .base_wrapper import BaseWrapper

__all__ = [
    "rtmo_multi_person_pose_estimation",
]

thisdir = Path(__file__).parent
root_path = thisdir.joinpath("../..")

# https://arxiv.org/pdf/2312.07526
# Reference: https://github.com/open-mmlab/mmpose/tree/main/projects/rtmo


class Split_Points(Enum):
    def __str__(self):
        return str(self.value)

    Backbone = "backbone"


@register_vision_model("rtmo_multi_person_pose_estimation")
class rtmo_multi_person_pose_estimation(BaseWrapper):
    def __init__(self, device: str, **kwargs):
        super().__init__(device)

        _path_prefix = (
            f"{root_path}"
            if kwargs["model_path_prefix"] == "default"
            else kwargs["model_path_prefix"]
        )
        self.model_info = {
            "cfg": f"{_path_prefix}/{kwargs['cfg']}",
            "weights": f"{_path_prefix}/{kwargs['weights']}",
        }

        cfg = Config.fromfile(self.model_info["cfg"])
        model_cfg = cfg["model"]
        log_processor_cfg = cfg.get("log_processor")
        default_scope = cfg.get("default_scope", "mmengine")
        assert default_scope == "mmpose"

        default_scope = DefaultScope.get_instance("mmpose", scope_name=default_scope)
        self.model = MODELS.build(model_cfg)
        self.test_cfg = model_cfg["test_cfg"]

        class dummy_data:
            metainfo = {"input_size": self.test_cfg["input_size"]}

        self.dummy_dsamples = [dummy_data]

        weights = torch.load(self.model_info["weights"], map_location="cpu")
        self.model.load_state_dict(
            weights["state_dict"],
            strict=True,
        )
        self.model.to(device).eval()
        self.backbone = self.model.extract_feat
        self.head = self.model.head

        self.supported_split_points = Split_Points
        assert "splits" in kwargs, "Split layer ids must be provided"
        self.split_id = str(kwargs["splits"]).lower()
        if self.split_id == str(self.supported_split_points.Backbone):
            self.split_layer_list = ["bb_p1", "bb_p2"]
        else:
            raise NotImplementedError

        self.features_at_splits = dict(
            zip(self.split_layer_list, [None] * len(self.split_layer_list))
        )

        if "logging_level" in kwargs:
            self.logger.level = kwargs["logging_level"]
            # logging.DEBUG

    @property
    def SPLIT_BACKBONE(self):
        return str(self.supported_split_points.Backbone)

    def input_to_features(self, x, device: str) -> Dict:
        """Computes deep features at the intermediate layer(s) all the way from the input"""

        self.model = self.model.to(device).eval()
        img = x[0]["image"].unsqueeze(0).to(device)
        input_size = tuple(img.shape[2:])

        if self.split_id == self.SPLIT_BACKBONE:
            output = self._input_to_feature_at_backbone(img)
        else:
            self.logger.error(f"Not supported split point {self.split_id}")
            raise NotImplementedError

        output["input_size"] = [input_size]
        return output

    def features_to_output(self, x: Dict, device: str):
        """Complete the downstream task from the intermediate deep features"""

        self.model = self.model.to(device).eval()

        if self.split_id == self.SPLIT_BACKBONE:
            return self._feature_at_backbone_to_output(
                x["data"], x["org_input_size"], x["input_size"]
            )
        else:
            self.logger.error(f"Not supported split points {self.split_id}")

        raise NotImplementedError

    @torch.no_grad()
    def _input_to_feature_at_backbone(self, x):
        """Computes and return feature at the backbone outputing two feature tensors all the way from the input"""

        features = self.backbone(x)
        assert len(self.features_at_splits) == len(features)

        for key, val in zip(self.features_at_splits.keys(), features):
            self.features_at_splits[key] = val

        return {"data": self.features_at_splits}

    @torch.no_grad()
    def _feature_at_backbone_to_output(
        self, x: Dict, org_img_size: Dict, input_img_size: List
    ):
        """
        performs  downstream task using the features from the output of the backbone network

        MMPOSE RTMO source codes are referenced for this function.
        <https://github.com/open-mmlab/mmpose/tree/main/projects/rtmo>

        Unnecessary parts for split inference are removed or modified properly.

        Please find the license statement in the downloaded original MMPOSE source codes or at here:
        <https://github.com/open-mmlab/mmpose?tab=Apache-2.0-1-ov-file#readme>

        """
        # must no use of data in self.features_at_splits

        assert len(self.features_at_splits) == len(x)
        features = tuple(x.values())
        preds = self.head.predict(features, self.dummy_dsamples, test_cfg=self.test_cfg)

        return preds

    @torch.no_grad()
    def forward(self, x):
        """Complete the downstream task with end-to-end manner all the way from the input"""

        self.model = self.model.to(self.device).eval()
        img = x["image"].unsqueeze(0).to(self.device)

        features = self.backbone(img)
        preds = self.head.predict(features, self.dummy_dsamples, test_cfg=self.test_cfg)

        return preds
