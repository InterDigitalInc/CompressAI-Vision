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

from collections import OrderedDict
from enum import Enum
from pathlib import Path
from typing import Dict, List

import torch

from torch import Tensor
from torch.nn import functional as F

from compressai_vision.registry import register_vision_model

from .base_wrapper import BaseWrapper

__all__ = [
    "fasterrcnn_mobilenet_v3_large_320_fpn",
    "lraspp_mobilenet_v3_large",
]

thisdir = Path(__file__).parent
root_path = thisdir.joinpath("../..")


def _extract_torchvision_images(x, device):
    if isinstance(x, torch.Tensor):
        if x.dim() == 3:
            return [x.to(device)]
        if x.dim() == 4:
            return [img.to(device) for img in x]
    if isinstance(x, list):
        images = []
        for item in x:
            image = item["image"] if isinstance(item, dict) else item
            if image.dim() == 4:
                assert image.shape[0] == 1
                image = image[0]
            images.append(image.to(device))
        return images
    raise TypeError(f"Unsupported input type for torchvision wrapper: {type(x)}")


def _resolve_weight_enum(enum_cls, weight_name):
    if weight_name is None:
        return None
    if str(weight_name).lower() in ("none", "false"):
        return None
    if str(weight_name).lower() in ("true", "default"):
        return enum_cls.DEFAULT
    return getattr(enum_cls, str(weight_name))


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


class TorchvisionDetectionSplitPoints(Enum):
    def __str__(self):
        return str(self.value)

    FeaturePyramidNetwork = "fpn"


@register_vision_model("fasterrcnn_mobilenet_v3_large_320_fpn")
class fasterrcnn_mobilenet_v3_large_320_fpn(BaseWrapper):
    def __init__(self, device: str, **kwargs):
        from torchvision.models.detection import (
            FasterRCNN_MobileNet_V3_Large_320_FPN_Weights,
            fasterrcnn_mobilenet_v3_large_320_fpn,
        )

        super().__init__(device)

        weights = _resolve_weight_enum(
            FasterRCNN_MobileNet_V3_Large_320_FPN_Weights,
            kwargs.get("weights", "DEFAULT"),
        )
        weights_backbone = kwargs.get("weights_backbone", None)
        if weights_backbone is not None:
            from torchvision.models import MobileNet_V3_Large_Weights

            weights_backbone = _resolve_weight_enum(
                MobileNet_V3_Large_Weights, weights_backbone
            )

        model_kwargs = {
            "weights": weights,
            "progress": bool(kwargs.get("progress", True)),
            "weights_backbone": weights_backbone,
        }
        for name in (
            "num_classes",
            "trainable_backbone_layers",
            "min_size",
            "max_size",
            "rpn_pre_nms_top_n_test",
            "rpn_post_nms_top_n_test",
            "rpn_score_thresh",
            "box_score_thresh",
            "box_nms_thresh",
            "box_detections_per_img",
        ):
            if name in kwargs and kwargs[name] is not None:
                model_kwargs[name] = kwargs[name]

        self.model = fasterrcnn_mobilenet_v3_large_320_fpn(**model_kwargs)
        self.model.to(device).eval()

        for param in self.model.parameters():
            param.requires_grad = False

        self.backbone = self.model.backbone
        self.rpn = self.model.rpn
        self.roi_heads = self.model.roi_heads
        self.transform = self.model.transform

        self.supported_split_points = TorchvisionDetectionSplitPoints
        self.split_id = str(kwargs["splits"]).lower()
        if self.split_id != str(self.supported_split_points.FeaturePyramidNetwork):
            raise NotImplementedError

        # Torchvision's MobileNetV3 FPN produces three maps in torchvision
        # 0.15.1: "0", "1", and "pool". Use neutral public names because this
        # is not a direct Detectron2 p2-p5 four-level pyramid.
        self.backbone_feature_keys = list(kwargs.get("backbone_feature_keys", [])) or [
            "0",
            "1",
            "pool",
        ]
        self.split_layer_list = list(kwargs.get("split_layer_names", [])) or [
            "fpn0",
            "fpn1",
            "fpn_pool",
        ]
        if len(self.split_layer_list) != len(self.backbone_feature_keys):
            raise ValueError(
                "split_layer_names and backbone_feature_keys must have the same length"
            )
        self.features_at_splits = dict(
            zip(self.split_layer_list, [None] * len(self.split_layer_list))
        )

        self.model_info = {
            "cfg": kwargs.get("cfg", "torchvision built-in configuration"),
            "weights": kwargs.get(
                "weights_url",
                getattr(weights, "url", "torchvision built-in weights"),
            ),
        }

    @property
    def SPLIT_FPN(self):
        return str(self.supported_split_points.FeaturePyramidNetwork)

    def input_to_features(self, x, device: str) -> Dict:
        self.model = self.model.to(device).eval()
        images = _extract_torchvision_images(x, device)
        transformed_images, _ = self.transform(images, None)
        features = self.backbone(transformed_images.tensors)
        if isinstance(features, Tensor):
            features = OrderedDict([("0", features)])

        self.features_at_splits = {
            public_key: features[torchvision_key]
            for public_key, torchvision_key in zip(
                self.split_layer_list, self.backbone_feature_keys
            )
        }
        return {
            "data": self.features_at_splits,
            "input_size": transformed_images.image_sizes,
        }

    def features_to_output(self, x: Dict, device: str):
        self.model = self.model.to(device).eval()
        return self._feature_pyramid_to_output(
            x["data"], x["org_input_size"], x["input_size"], device
        )

    @torch.no_grad()
    def _feature_pyramid_to_output(
        self, x: Dict, org_img_size: Dict, input_img_size: List, device
    ):
        """
        Finish torchvision Faster R-CNN from FPN features.

        This follows torchvision's GeneralizedRCNN forward path after the
        backbone. See the torchvision BSD-3-Clause license at
        https://github.com/pytorch/vision/blob/main/LICENSE.
        """
        from detectron2.structures import Boxes, Instances
        from torchvision.models.detection.image_list import ImageList

        features = OrderedDict(
            (torchvision_key, x[public_key])
            for public_key, torchvision_key in zip(
                self.split_layer_list, self.backbone_feature_keys
            )
        )

        image_size = _as_hw(input_img_size)
        dummy_image = torch.zeros((3, image_size[0], image_size[1]), device=device)
        batched = self.transform.batch_images([dummy_image])
        images = ImageList(batched, [tuple(image_size)])

        proposals, _ = self.rpn(images, features, None)
        detections, _ = self.roi_heads(features, proposals, images.image_sizes, None)
        detections = self.transform.postprocess(
            detections,
            images.image_sizes,
            [(org_img_size["height"], org_img_size["width"])],
        )

        processed_results = []
        for result in detections:
            boxes = Boxes(result["boxes"])
            scores = result["scores"]
            labels = result["labels"]
            instances = Instances(
                image_size=(org_img_size["height"], org_img_size["width"])
            )
            instances.set("pred_boxes", boxes)
            instances.set("scores", scores)
            instances.set("pred_classes", labels.to(dtype=torch.int64))
            instances.set("pred_classes_dataset_id", labels.to(dtype=torch.int64))
            processed_results.append({"instances": instances})

        return processed_results

    @torch.no_grad()
    def get_input_size(self, x):
        images = _extract_torchvision_images(x, self.device)
        transformed_images, _ = self.transform(images, None)
        return transformed_images.image_sizes

    @torch.no_grad()
    def forward(self, x):
        images = _extract_torchvision_images([x], self.device)
        detections = self.model(images)
        from detectron2.structures import Boxes, Instances

        processed_results = []
        for result in detections:
            org_h, org_w = x["height"], x["width"]
            instances = Instances(image_size=(org_h, org_w))
            instances.set("pred_boxes", Boxes(result["boxes"]))
            instances.set("scores", result["scores"])
            labels = result["labels"]
            instances.set("pred_classes", labels.to(dtype=torch.int64))
            instances.set("pred_classes_dataset_id", labels.to(dtype=torch.int64))
            processed_results.append({"instances": instances})
        return processed_results

    def calc_complexity(self, mode, input, data=None):
        raise NotImplementedError(
            "Complexity calculation is not implemented for torchvision Faster R-CNN"
        )


class TorchvisionLRASPPSplitPoints(Enum):
    def __str__(self):
        return str(self.value)

    Backbone = "backbone"
    Logits = "logits"


@register_vision_model("lraspp_mobilenet_v3_large")
class lraspp_mobilenet_v3_large(BaseWrapper):
    def __init__(self, device: str, **kwargs):
        from torchvision.models.segmentation import (
            LRASPP_MobileNet_V3_Large_Weights,
            lraspp_mobilenet_v3_large,
        )

        super().__init__(device)

        weights = _resolve_weight_enum(
            LRASPP_MobileNet_V3_Large_Weights, kwargs.get("weights", "DEFAULT")
        )
        weights_backbone = kwargs.get("weights_backbone", None)
        if weights_backbone is not None:
            from torchvision.models import MobileNet_V3_Large_Weights

            weights_backbone = _resolve_weight_enum(
                MobileNet_V3_Large_Weights, weights_backbone
            )

        model_kwargs = {
            "weights": weights,
            "progress": bool(kwargs.get("progress", True)),
            "weights_backbone": weights_backbone,
        }
        if "num_classes" in kwargs and kwargs["num_classes"] is not None:
            model_kwargs["num_classes"] = kwargs["num_classes"]

        self.model = lraspp_mobilenet_v3_large(**model_kwargs)
        self.model.to(device).eval()
        for param in self.model.parameters():
            param.requires_grad = False

        self.preprocess = None
        if weights is not None and kwargs.get("use_weights_transforms", True):
            self.preprocess = weights.transforms()

        self.backbone = self.model.backbone
        self.classifier = self.model.classifier
        self.supported_split_points = TorchvisionLRASPPSplitPoints
        self.split_id = str(kwargs["splits"]).lower()

        if self.split_id == str(self.supported_split_points.Backbone):
            self.split_layer_list = ["low", "high"]
        elif self.split_id == str(self.supported_split_points.Logits):
            self.split_layer_list = ["logits"]
        else:
            raise NotImplementedError

        self.features_at_splits = dict(
            zip(self.split_layer_list, [None] * len(self.split_layer_list))
        )
        self.model_info = {
            "cfg": kwargs.get("cfg", "torchvision built-in configuration"),
            "weights": kwargs.get(
                "weights_url",
                getattr(weights, "url", "torchvision built-in weights"),
            ),
        }

    @property
    def SPLIT_BACKBONE(self):
        return str(self.supported_split_points.Backbone)

    @property
    def SPLIT_LOGITS(self):
        return str(self.supported_split_points.Logits)

    def input_to_features(self, x, device: str) -> Dict:
        self.model = self.model.to(device).eval()
        images = _extract_torchvision_images(x, device)
        assert len(images) == 1, "LR-ASPP wrapper currently supports batch size 1"
        img = images[0]
        if self.preprocess is not None:
            img = self.preprocess(img)
        img = img.unsqueeze(0)
        input_size = tuple(img.shape[-2:])

        if self.split_id == self.SPLIT_BACKBONE:
            features = self.backbone(img)
            self.features_at_splits = {
                key: features[key] for key in self.split_layer_list
            }
        elif self.split_id == self.SPLIT_LOGITS:
            features = self.backbone(img)
            logits = self.classifier(features)
            self.features_at_splits = {"logits": logits}
        else:
            raise NotImplementedError

        return {"data": self.features_at_splits, "input_size": [input_size]}

    def features_to_output(self, x: Dict, device: str):
        self.model = self.model.to(device).eval()
        return self._features_to_segmentation(x["data"], x["input_size"], device)

    @torch.no_grad()
    def _features_to_segmentation(self, x: Dict, input_img_size: List, device):
        """
        Finish torchvision LR-ASPP from either backbone features or logits.

        Backbone split sends the MobileNetV3 "low" and "high" feature maps.
        Logits split sends the classifier output before the final upsample.
        """
        input_size = _as_hw(input_img_size)
        if self.split_id == self.SPLIT_BACKBONE:
            logits = self.classifier(
                OrderedDict((key, x[key].to(device)) for key in self.split_layer_list)
            )
        elif self.split_id == self.SPLIT_LOGITS:
            logits = x["logits"].to(device)
        else:
            raise NotImplementedError

        out = F.interpolate(
            logits, size=input_size, mode="bilinear", align_corners=False
        )
        return [OrderedDict(out=out[0])]

    @torch.no_grad()
    def get_input_size(self, x):
        images = _extract_torchvision_images(x, self.device)
        assert len(images) == 1
        img = images[0]
        if self.preprocess is not None:
            img = self.preprocess(img)
        return [tuple(img.shape[-2:])]

    @torch.no_grad()
    def forward(self, x):
        images = _extract_torchvision_images([x], self.device)
        assert len(images) == 1
        img = images[0]
        if self.preprocess is not None:
            img = self.preprocess(img)
        return self.model(img.unsqueeze(0))

    def calc_complexity(self, mode, input, data=None):
        raise NotImplementedError(
            "Complexity calculation is not implemented for torchvision LR-ASPP"
        )
