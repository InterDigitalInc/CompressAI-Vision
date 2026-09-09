# Copyright (c) 2026, InterDigital Communications, Inc
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

import json

from enum import Enum
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch

from compressai_vision.registry import register_vision_model

from .base_wrapper import BaseWrapper

__all__ = [
    "efficientvit_sam_l0",
    "efficientvit_sam_l1",
    "efficientvit_sam_l2",
    "efficientvit_sam_xl0",
    "efficientvit_sam_xl1",
]

thisdir = Path(__file__).parent
root_path = thisdir.joinpath("../..")


def mask_to_bbx(mask):
    mask = mask.detach().cpu()
    mask = np.squeeze(np.array(mask))
    rows, cols = np.where(mask)
    if len(rows) == 0 or len(cols) == 0:
        return [0, 0, 0, 0]
    return [
        cols.min(),
        rows.min(),
        cols.max(),
        rows.max(),
    ]


def bbox_xywh_to_xyxy(bbox):
    return [bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]]


def _as_hw(size):
    while isinstance(size, list) and len(size) == 1:
        size = size[0]
    if isinstance(size, tuple) and len(size) == 1:
        size = size[0]
    if isinstance(size, torch.Size):
        size = tuple(size)
    if len(size) != 2:
        raise ValueError(f"Expected an image size pair, got {size}")
    return int(size[0]), int(size[1])


class Split_Points(Enum):
    def __str__(self):
        return str(self.value)

    ImageEncoder = "imgenc"


class EfficientViTSAM(BaseWrapper):
    efficientvit_model_name = None

    def __init__(self, device: str, **kwargs):
        from efficientvit.models.efficientvit.sam import EfficientViTSamPredictor

        super().__init__(device)

        _path_prefix = (
            f"{root_path}"
            if kwargs["model_path_prefix"] == "default"
            else kwargs["model_path_prefix"]
        )
        weight_url = kwargs.get("weights", None)
        if weight_url is not None and str(weight_url).lower() not in (
            "none",
            "false",
        ):
            if not str(weight_url).startswith(("http://", "https://")):
                weight_url = f"{_path_prefix}/{weight_url}"
        else:
            weight_url = None

        self.model_name = kwargs.get("model_name", self.efficientvit_model_name)
        self.model = self._create_model(
            self.model_name, bool(kwargs.get("pretrained", True)), weight_url
        )
        self.model.to(device).eval()
        for param in self.model.parameters():
            param.requires_grad = False

        self.image_encoder = self.model.image_encoder
        self.prompt_encoder = self.model.prompt_encoder
        self.head = self.model.mask_decoder
        self.predictor = EfficientViTSamPredictor(self.model)
        self.prompt_type = kwargs.get("prompt_type", "point_file")
        self.source_json_file = kwargs.get("source_json_file", None)
        self.prompt_score_threshold = kwargs.get("prompt_score_threshold", None)
        self.max_prompts_per_image = kwargs.get("max_prompts_per_image", None)
        self._detector_prompts = self._load_detector_prompts(self.source_json_file)
        self.supported_split_points = Split_Points

        self.split_id = str(kwargs["splits"]).lower()
        if self.split_id != str(self.supported_split_points.ImageEncoder):
            raise NotImplementedError

        self.split_layer_list = ["imgenc"]
        self.features_at_splits = {"imgenc": None}
        self.model_info = {
            "cfg": kwargs.get("cfg", "efficientvit built-in configuration"),
            "weights": weight_url or "efficientvit built-in weight URL",
        }

    @staticmethod
    def _create_model(model_name, pretrained, weight_url):
        try:
            from efficientvit.sam_model_zoo import create_efficientvit_sam_model

            full_name = (
                model_name
                if str(model_name).startswith("efficientvit-sam-")
                else f"efficientvit-sam-{model_name}"
            )
            return create_efficientvit_sam_model(
                full_name,
                pretrained=pretrained,
                weight_url=weight_url,
            )
        except ImportError:
            from efficientvit.sam_model_zoo import create_sam_model

            short_name = str(model_name).replace("efficientvit-sam-", "")
            return create_sam_model(
                short_name,
                pretrained=pretrained,
                weight_url=weight_url,
            )

    def _load_detector_prompts(self, source_json_file):
        if self.prompt_type != "box_from_detector":
            return {}
        if source_json_file is None:
            raise ValueError(
                "EfficientViT-SAM prompt_type=box_from_detector requires "
                "source_json_file"
            )
        source_path = Path(source_json_file)
        if not source_path.is_absolute():
            source_path = root_path / source_path

        with open(source_path, "r", encoding="utf-8") as f:
            detections = json.load(f)

        prompts = {}
        for det in detections:
            score = det.get("score", 1.0)
            if (
                self.prompt_score_threshold is not None
                and score < self.prompt_score_threshold
            ):
                continue
            image_id = int(det["image_id"])
            prompts.setdefault(image_id, []).append(
                {
                    "box": bbox_xywh_to_xyxy(det["bbox"]),
                    "class": int(det["category_id"]),
                    "score": float(score),
                }
            )

        for image_id in prompts:
            prompts[image_id].sort(key=lambda item: item["score"], reverse=True)
            if self.max_prompts_per_image is not None:
                prompts[image_id] = prompts[image_id][: self.max_prompts_per_image]

        return prompts

    @property
    def SPLIT_IMGENC(self):
        return str(self.supported_split_points.ImageEncoder)

    @staticmethod
    def prompt_inputs(file_name):
        prompt_link = file_name.replace("/images/", "/prompts/").replace(".jpg", ".txt")
        prompts = []
        object_classes = []
        with open(prompt_link, "r") as f:
            for line in f:
                parts = line.strip().split()
                prompts.append(list(map(int, parts[:2])))
                object_classes.append(int(parts[-1]))
        return prompts, object_classes

    def _prompt_items(self, sample_info):
        if self.prompt_type == "box_from_detector":
            image_id = sample_info.get("image_id", None)
            if image_id is None:
                image_id = int(Path(sample_info["file_name"]).stem)
            return self._detector_prompts.get(int(image_id), [])

        prompts, object_classes = self.prompt_inputs(sample_info["file_name"])
        return [
            {
                "point": prompt,
                "class": object_class,
                "score": None,
            }
            for prompt, object_class in zip(prompts, object_classes)
        ]

    def _prepare_image(self, item, device):
        img = item["image"] if isinstance(item, dict) else item
        if isinstance(img, np.ndarray):
            self.predictor.set_image(img)
            return (
                self.predictor.features.to(device),
                tuple(self.predictor.input_size),
            )

        if isinstance(img, torch.Tensor):
            tensor = img.to(device)
            if tensor.dim() == 4:
                assert tensor.shape[0] == 1
                tensor = tensor[0]
            if tensor.dim() == 3:
                image = tensor.detach().cpu()
                if image.shape[0] in (1, 3):
                    image = image.permute(1, 2, 0)
                if image.is_floating_point() and image.max() <= 1:
                    image = image * 255
                image = image.clamp(0, 255).to(torch.uint8).numpy()
                self.predictor.set_image(image)
                return (
                    self.predictor.features.to(device),
                    tuple(self.predictor.input_size),
                )

        raise TypeError(
            "EfficientViT-SAM expects a numpy RGB image or a 3D/4D torch tensor"
        )

    def input_to_features(self, x: list, device: str) -> Dict:
        self.model = self.model.to(device).eval()
        assert isinstance(x, list) and len(x) == 1
        if self.split_id != self.SPLIT_IMGENC:
            raise NotImplementedError
        return self._input_to_image_encoder(x, device)

    def features_to_output(self, x: Dict, device: str):
        self.model = self.model.to(device).eval()
        if self.split_id == self.SPLIT_IMGENC:
            return self._image_encoder_to_output(
                x["data"],
                x["org_input_size"],
                x["input_size"],
                self._prompt_items(x),
                device,
            )
        raise NotImplementedError

    @torch.no_grad()
    def _input_to_image_encoder(self, x, device):
        prepared, input_size = self._prepare_image(x[0], device)
        self.features_at_splits["imgenc"] = prepared
        return {"data": self.features_at_splits, "input_size": list(input_size)}

    @torch.no_grad()
    def get_input_size(self, x):
        _, input_size = self._prepare_image(x[0], self.device)
        return list(input_size)

    @torch.no_grad()
    def _image_encoder_to_output(
        self,
        x: Dict,
        org_img_size: Dict,
        input_img_size: List,
        prompt_items: List[Dict],
        device,
    ):
        instances_list = []
        self.predictor.features = x["imgenc"].to(device)
        self.predictor.original_size = (
            org_img_size["height"],
            org_img_size["width"],
        )
        self.predictor.input_size = _as_hw(input_img_size)
        self.predictor.is_image_set = True

        from detectron2.structures import Boxes, Instances

        for prompt_item in prompt_items:
            if "box" in prompt_item:
                masks, iou_pred, _ = self.predictor.predict(
                    point_coords=None,
                    point_labels=None,
                    box=np.array(prompt_item["box"], dtype=np.float32),
                    multimask_output=True,
                )
                best_idx = int(np.asarray(iou_pred).argmax())
                mask = masks[best_idx]
                iou_score = float(np.asarray(iou_pred).reshape(-1)[best_idx])
            else:
                masks, iou_pred, _ = self.predictor.predict(
                    point_coords=np.array([prompt_item["point"]], dtype=np.float32),
                    point_labels=np.array([1], dtype=np.int32),
                    multimask_output=False,
                )
                mask = masks[0]
                iou_score = float(np.asarray(iou_pred).reshape(-1)[0])

            mask = torch.as_tensor(mask, device=device)
            boxes = Boxes(torch.tensor(np.array([mask_to_bbx(mask)]), device=device))
            score = prompt_item.get("score", None)
            if score is None:
                score = iou_score
            scores = torch.tensor([score], dtype=torch.float32, device=device)
            classes = torch.tensor(
                [prompt_item["class"]], dtype=torch.int64, device=device
            )

            instances = Instances(
                image_size=(org_img_size["height"], org_img_size["width"])
            )
            instances.set("pred_boxes", boxes)
            instances.set("scores", scores)
            instances.set("pred_classes", classes)
            instances.set("pred_classes_dataset_id", classes)
            instances.set("pred_masks", mask[None, ...])
            instances_list.append(instances)

        if instances_list:
            instances = Instances.cat(instances_list)
        else:
            instances = Instances(
                image_size=(org_img_size["height"], org_img_size["width"])
            )
            empty_boxes = torch.empty((0, 4), dtype=torch.float32, device=device)
            empty_scores = torch.empty((0,), dtype=torch.float32, device=device)
            empty_classes = torch.empty((0,), dtype=torch.int64, device=device)
            empty_masks = torch.empty(
                (0, org_img_size["height"], org_img_size["width"]),
                dtype=torch.bool,
                device=device,
            )
            instances.set("pred_boxes", Boxes(empty_boxes))
            instances.set("scores", empty_scores)
            instances.set("pred_classes", empty_classes)
            instances.set("pred_classes_dataset_id", empty_classes)
            instances.set("pred_masks", empty_masks)

        return [{"instances": instances}]

    @torch.no_grad()
    def forward(self, x):
        enc_res = self._input_to_image_encoder([x], self.device)
        return self._image_encoder_to_output(
            enc_res["data"],
            {"height": x["height"], "width": x["width"]},
            enc_res["input_size"],
            self._prompt_items(x),
            device=self.device,
        )

    def calc_complexity(self, mode, input, data=None):
        raise NotImplementedError(
            "Complexity calculation is not implemented for EfficientViT-SAM"
        )


@register_vision_model("efficientvit_sam_l0")
class efficientvit_sam_l0(EfficientViTSAM):
    efficientvit_model_name = "l0"


@register_vision_model("efficientvit_sam_l1")
class efficientvit_sam_l1(EfficientViTSAM):
    efficientvit_model_name = "l1"


@register_vision_model("efficientvit_sam_l2")
class efficientvit_sam_l2(EfficientViTSAM):
    efficientvit_model_name = "l2"


@register_vision_model("efficientvit_sam_xl0")
class efficientvit_sam_xl0(EfficientViTSAM):
    efficientvit_model_name = "xl0"


@register_vision_model("efficientvit_sam_xl1")
class efficientvit_sam_xl1(EfficientViTSAM):
    efficientvit_model_name = "xl1"
