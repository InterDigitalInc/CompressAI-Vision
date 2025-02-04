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
from enum import Enum
from pathlib import Path
from typing import Dict, List

import torch
from yolox.exp import get_exp
from yolox.utils import postprocess

from compressai_vision.registry import register_vision_model

from .base_wrapper import BaseWrapper

__all__ = [
    "yolox_darknet53",
]

thisdir = Path(__file__).parent
root_path = thisdir.joinpath("../..")


class Split_Points(Enum):
    def __str__(self):
        return str(self.value)

    Layer13_Single = "l13"
    Layer37_Single = "l37"


@register_vision_model("yolox_darknet53")
class yolox_darknet53(BaseWrapper):
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

        self.num_classes = kwargs["num_classes"]
        self.conf_thres = kwargs["conf_thres"]
        self.nms_thres = kwargs["nms_thres"]

        self.supported_split_points = Split_Points

        exp = get_exp(exp_file=None, exp_name="yolov3")

        self.model = exp.get_model()
        # check with exp.output_dir

        assert "splits" in kwargs, "Split layer ids must be provided"
        self.split_id = str(kwargs["splits"]).lower()
        if self.split_id == str(self.supported_split_points.Layer13_Single):
            self.split_layer_list = ["l13"]
        elif self.split_id == str(self.supported_split_points.Layer37_Single):
            self.split_layer_list = ["l37"]
        else:
            raise NotImplementedError

        self.features_at_splits = dict(
            zip(self.split_layer_list, [None] * len(self.split_layer_list))
        )

        self.model.load_state_dict(
            torch.load(self.model_info["weights"], map_location="cpu")["model"],
            strict=False,
        )
        self.model.to(device).eval()

        self.yolo_fpn = self.model.backbone
        self.backbone = self.yolo_fpn.backbone
        self.head = self.model.head

        if "logging_level" in kwargs:
            self.logger.level = kwargs["logging_level"]
            # logging.DEBUG

    @property
    def SPLIT_L13(self):
        return str(self.supported_split_points.Layer13_Single)

    @property
    def SPLIT_L37(self):
        return str(self.supported_split_points.Layer37_Single)

    def input_to_features(self, x, device: str) -> Dict:
        """Computes deep features at the intermediate layer(s) all the way from the input"""

        self.model = self.model.to(device).eval()
        img = x[0]["image"].unsqueeze(0).to(device)
        input_size = tuple(img.shape[2:])

        if self.split_id == self.SPLIT_L13:
            output = self._input_to_feature_at_l13(img)
        elif self.split_id == self.SPLIT_L37:
            output = self._input_to_feature_at_l37(img)
        else:
            self.logger.error(f"Not supported split point {self.split_id}")
            raise NotImplementedError

        output["input_size"] = [input_size]
        return output

    def features_to_output(self, x: Dict, device: str):
        """Complete the downstream task from the intermediate deep features"""

        self.model = self.model.to(device).eval()

        if self.split_id == self.SPLIT_L13:
            return self._feature_at_l13_to_output(
                x["data"], x["org_input_size"], x["input_size"]
            )
        elif self.split_id == self.SPLIT_L37:
            return self._feature_at_l37_to_output(
                x["data"], x["org_input_size"], x["input_size"]
            )
        else:
            self.logger.error(f"Not supported split points {self.split_id}")

        raise NotImplementedError

    @torch.no_grad()
    def _input_to_feature_at_l13(self, x):
        """Computes and return feature at layer 13 with leaky relu all the way from the input"""

        y = self.backbone.stem(x)
        y = self.backbone.dark2(y)
        self.features_at_splits[self.SPLIT_L13] = self.backbone.dark3[0](y)

        return {"data": self.features_at_splits}

    @torch.no_grad()
    def _input_to_feature_at_l37(self, x):
        """Computes and return feature at layer 37 with 11th residual layer output all the way from the input"""

        y = self.backbone.stem(x)
        y = self.backbone.dark2(y)
        y = self.backbone.dark3(y)
        self.features_at_splits[self.SPLIT_L37] = y

        return {"data": self.features_at_splits}

    @torch.no_grad()
    def _feature_at_l13_to_output(
        self, x: Dict, org_img_size: Dict, input_img_size: List
    ):
        """
        performs  downstream task using the features from layer 13

        YOLOX source codes are referenced for this function.
        <https://github.com/Megvii-BaseDetection/YOLOX/yolox/data/data_augment.py>

        Unnecessary parts for split inference are removed or modified properly.

        Please find the license statement in the downloaded original YOLOX source codes or at here:
        <https://github.com/Megvii-BaseDetection/YOLOX?tab=Apache-2.0-1-ov-file#readme>

        """

        y = x[self.SPLIT_L13]
        for proc_module in self.backbone.dark3[1:]:
            y = proc_module(y)

        fp_lvl2 = y
        fp_lvl1 = self.backbone.dark4(fp_lvl2)
        fp_lvl0 = self.backbone.dark5(fp_lvl1)

        # yolo branch 1
        b1_in = self.yolo_fpn.out1_cbl(fp_lvl0)
        b1_in = self.yolo_fpn.upsample(b1_in)
        b1_in = torch.cat([b1_in, fp_lvl1], 1)
        fp_lvl1 = self.yolo_fpn.out1(b1_in)

        # yolo branch 2
        b2_in = self.yolo_fpn.out2_cbl(fp_lvl1)
        b2_in = self.yolo_fpn.upsample(b2_in)
        b2_in = torch.cat([b2_in, fp_lvl2], 1)
        fp_lvl2 = self.yolo_fpn.out2(b2_in)

        outputs = self.head((fp_lvl2, fp_lvl1, fp_lvl0))

        pred = postprocess(outputs, self.num_classes, self.conf_thres, self.nms_thres)

        return pred

    @torch.no_grad()
    def _feature_at_l37_to_output(
        self, x: Dict, org_img_size: Dict, input_img_size: List
    ):
        """
        performs  downstream task using the features from layer 37

        YOLOX source codes are referenced for this function.
        <https://github.com/Megvii-BaseDetection/YOLOX/yolox/data/data_augment.py>

        Unnecessary parts for split inference are removed or modified properly.

        Please find the license statement in the downloaded original YOLOX source codes or at here:
        <https://github.com/Megvii-BaseDetection/YOLOX?tab=Apache-2.0-1-ov-file#readme>

        """

        fp_lvl2 = x[self.SPLIT_L37]
        fp_lvl1 = self.backbone.dark4(fp_lvl2)
        fp_lvl0 = self.backbone.dark5(fp_lvl1)

        # yolo branch 1
        b1_in = self.yolo_fpn.out1_cbl(fp_lvl0)
        b1_in = self.yolo_fpn.upsample(b1_in)
        b1_in = torch.cat([b1_in, fp_lvl1], 1)
        fp_lvl1 = self.yolo_fpn.out1(b1_in)

        # yolo branch 2
        b2_in = self.yolo_fpn.out2_cbl(fp_lvl1)
        b2_in = self.yolo_fpn.upsample(b2_in)
        b2_in = torch.cat([b2_in, fp_lvl2], 1)
        fp_lvl2 = self.yolo_fpn.out2(b2_in)

        outputs = self.head((fp_lvl2, fp_lvl1, fp_lvl0))

        pred = postprocess(outputs, self.num_classes, self.conf_thres, self.nms_thres)

        return pred

    @torch.no_grad()
    def forward(self, x):
        """Complete the downstream task with end-to-end manner all the way from the input"""

        self.model = self.model.to(self.device).eval()
        img = x["image"].unsqueeze(0).to(self.device)

        fpn_out = self.yolo_fpn(img)
        outputs = self.head(fpn_out)

        pred = postprocess(outputs, self.num_classes, self.conf_thres, self.nms_thres)

        return pred
