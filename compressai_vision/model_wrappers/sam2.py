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

import base64
import csv
import os

from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Tuple, Union

import cv2
import numpy as np
import pandas
import torch

from torch.nn import functional as F

from compressai_vision.registry import register_vision_model

from .base_wrapper import BaseWrapper
from .sam import Boxes, mask_to_bbx

# sam = sam_model_registry["vit_h"](checkpoint="/t/vic/hevc_simulations/rosen/build/compressai13_sam/weights/sam/sam_vit_h_4b8939.pth")
# predictor = SamPredictor(sam)
# mask_generator = SamAutomaticMaskGenerator(sam, points_per_batch=16)


__all__ = [
    "sam2_hiera_image_model",
    "sam2_hiera_video_model",
]

thisdir = Path(__file__).parent
root_path = thisdir.joinpath("../..")


class Split_Points(Enum):
    def __str__(self):
        return str(self.value)

    ImageEncoder = "backbone"  # features output from neck.3.bias


class SAM2(BaseWrapper):
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

    @property
    def SPLIT_IMGENC(self):
        return str(self.supported_split_points.ImageEncoder)

    @staticmethod
    def prompt_inputs(file_name):
        # [TODO] should be improved...
        prompt_link = file_name.replace("/images/", "/prompts/").replace(".jpg", ".txt")

        with open(prompt_link, "r") as f:
            line = f.readline()
            # first_two = list(map(int, line.strip().split()[:2]))
            parts = line.strip().split()
            prompts = list(map(int, parts[:2]))
            object_classes = [int(line.strip().split()[-1])]

        return prompts, object_classes

    def input_to_features(self, x: list, device: str) -> Dict:
        """Computes deep features at the intermediate layer(s) all the way from the input"""
        self.model.model = self.model.model.to(device).eval()
        assert isinstance(x, list) and len(x) == 1

        if self.split_id == self.SPLIT_IMGENC:
            return self._input_to_image_encoder(x, device)
        else:
            self.logger.error(f"Not supported split point {self.split_id}")

        raise NotImplementedError

    def features_to_output(self, x: Dict, device: str):
        """Complete the downstream task from the intermediate deep features"""

        self.model.model = self.model.model.to(device).eval()

        if self.split_id == self.SPLIT_IMGENC:
            assert "file_name" in x

            prompts, object_classes = self.prompt_inputs(x["file_name"])

            return self._image_encoder_to_output(
                x["data"],
                x["org_input_size"],
                x["input_size"],
                prompts,
                object_classes,
                device,
            )
        else:
            self.logger.error(f"Not supported split points {self.split_id}")

        raise NotImplementedError

    @torch.no_grad()
    def _input_to_image_encoder(self, x, device):
        """Computes and return encoded image all the way from the input"""
        assert len(x) == 1

        img = x[0]["image"]
        assert isinstance(img, np.ndarray)
        assert len(img.shape) == 3
        input_size = list(img.shape[:2])

        self.image_encoder(img)

        feature = {
            "img_embed": self.model._features["image_embed"],
            "high_res_feats1": self.model._features["high_res_feats"][0],
            "high_res_feats2": self.model._features["high_res_feats"][1],
        }

        return {
            "data": feature,
            "input_size": input_size,
        }

    @torch.no_grad()
    def get_input_size(self, x):
        """Computes input image size to the network"""
        # TODO

        image_sizes = [x[0]["height"], x[0]["width"]]
        return image_sizes  # [1024, 1024]

    @torch.no_grad()
    def _image_encoder_to_output(
        self,
        x: Dict,
        org_img_size: Dict,
        input_img_size: List,
        prompts: List,
        object_classes: List,
        device,
    ):
        """
        performs  downstream task using the encoded image feature

        """
        # print("prompts object_classes", prompts,  object_classes)

        input_points = [prompts]  # [[469, 295]] #prompts["points"]
        input_points = np.array(input_points)
        input_labels = np.array([1])
        masks, iou_pred, low_res_masks = self.model.predict(
            point_coords=input_points, point_labels=input_labels, multimask_output=False
        )

        # mask_threshold = 0.0

        assert len(masks) == len(iou_pred)

        from detectron2.structures import Instances

        # post process result
        processed_results = []
        boxes = mask_to_bbx(masks[0])
        boxes = Boxes(torch.tensor(np.array([boxes])))
        scores = torch.tensor([iou_pred[0]])
        classes = torch.tensor(object_classes)

        # Create an instance
        instances = Instances(image_size=(input_img_size[0], input_img_size[1]))
        instances.set("pred_boxes", boxes)
        instances.set("scores", scores)
        instances.set("pred_classes", classes)
        instances.set("pred_masks", masks)

        processed_results.append({"instances": instances})

        return processed_results

    @torch.no_grad()
    def forward(self, x):
        """Complete the downstream task with end-to-end manner all the way from the input"""
        # test
        enc_res = self._input_to_image_encoder([x], self.device)

        # suppose that the order of keys and values is matched
        enc_res["data"] = {
            k: v.to(device=self.device)
            for k, v in zip(self.split_layer_list, enc_res["data"].values())
        }

        prompts, object_classes = self.prompt_inputs(x["file_origin"])

        dec_res = self._image_encoder_to_output(
            enc_res["data"],
            {"height": x["height"], "width": x["width"]},
            enc_res["input_size"],
            prompts,
            object_classes,
            device=self.device,
        )

        return dec_res


@register_vision_model("sam2_hiera_image_model")
class sam2_hiera_image_model(SAM2):
    def __init__(self, device: str, **kwargs):
        super().__init__(device, **kwargs)

        from hydra.utils import instantiate
        from omegaconf import OmegaConf
        from sam2.build_sam import _load_checkpoint, build_sam2
        from sam2.sam2_image_predictor import SAM2ImagePredictor

        sam2_cfg_extra = [
            "model.sam_mask_decoder_extra_args.dynamic_multimask_via_stability=true",
            "model.sam_mask_decoder_extra_args.dynamic_multimask_stability_delta=0.05",
            "model.sam_mask_decoder_extra_args.dynamic_multimask_stability_thresh=0.98",
        ]
        sam2_base_cfg = OmegaConf.load(self.model_info["cfg"])
        overrides_cfg = OmegaConf.from_dotlist(sam2_cfg_extra)
        sam2_final_cfg = OmegaConf.merge(sam2_base_cfg, overrides_cfg)
        OmegaConf.resolve(sam2_final_cfg)
        model = instantiate(sam2_final_cfg.model, _recursive_=True)
        _load_checkpoint(model, self.model_info["weights"])

        model = model.to(device).eval()
        self.model = SAM2ImagePredictor(model)
        self.image_encoder = self.model.set_image

        self.supported_split_points = Split_Points

        assert "splits" in kwargs, "Split layer ids must be provided"
        self.split_id = str(kwargs["splits"]).lower()

        self.split_layer_list = ["img_embed", "high_res_feats1", "high_res_feats2"]

        self.features_at_splits = dict(
            zip(self.split_layer_list, [None] * len(self.split_layer_list))
        )


@register_vision_model("sam2_hiera_video_model")
class sam2_hiera_video_model(SAM2):
    def __init__(self, device: str, **kwargs):
        super().__init__(device, **kwargs)

        raise NotImplementedError
