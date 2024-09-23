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

from enum import Enum
from pathlib import Path
from typing import Dict, List

import torch
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.modeling import build_model
from detectron2.structures import ImageList

from compressai_vision.registry import register_vision_model

from .base_wrapper import BaseWrapper

__all__ = [
    "faster_rcnn_X_101_32x8d_FPN_3x",
    "mask_rcnn_X_101_32x8d_FPN_3x",
    "faster_rcnn_R_50_FPN_3x",
    "mask_rcnn_R_50_FPN_3x",
]

thisdir = Path(__file__).parent
root_path = thisdir.joinpath("../..")


class Split_Points(Enum):
    def __str__(self):
        return str(self.value)

    FeaturePyramidNetwork = "fpn"
    C2 = "c2"
    Res2 = "r2"


class Rcnn_R_50_X_101_FPN(BaseWrapper):
    def __init__(self, device: str, **kwargs):
        super().__init__(device)

        self._cfg = get_cfg()
        self._cfg.MODEL.DEVICE = device
        _path_prefix = (
            f"{root_path}"
            if kwargs["model_path_prefix"] == "default"
            else kwargs["model_path_prefix"]
        )
        self._cfg.merge_from_file(f"{_path_prefix}/{kwargs['cfg']}")

        self.model = build_model(self._cfg).to(device).eval()

        self.backbone = self.model.backbone
        self.top_block = self.model.backbone.top_block
        self.proposal_generator = self.model.proposal_generator
        self.roi_heads = self.model.roi_heads
        self.postprocess = self.model._postprocess
        DetectionCheckpointer(self.model).load(f"{_path_prefix}/{kwargs['weights']}")

        self.model_info = {"cfg": kwargs["cfg"], "weights": kwargs["weights"]}

        self.supported_split_points = Split_Points

        assert "splits" in kwargs, "Split layer ids must be provided"
        self.split_id = str(kwargs["splits"]).lower()

        if self.split_id == str(self.supported_split_points.FeaturePyramidNetwork):
            self.split_layer_list = ["p2", "p3", "p4", "p5"]
        elif self.split_id == str(self.supported_split_points.C2):
            self.split_layer_list = ["c2", "c3", "c4", "c5"]
        elif self.split_id == str(self.supported_split_points.Res2):
            self.split_layer_list = ["r2"]
        else:
            raise NotImplementedError

        self.features_at_splits = dict(
            zip(self.split_layer_list, [None] * len(self.split_layer_list))
        )

        assert self.top_block is not None
        assert self.proposal_generator is not None

    @property
    def SPLIT_FPN(self):
        return str(self.supported_split_points.FeaturePyramidNetwork)

    @property
    def SPLIT_C2(self):
        return str(self.supported_split_points.C2)

    @property
    def SPLIT_R2(self):
        return str(self.supported_split_points.Res2)

    @property
    def size_divisibility(self):
        return self.backbone.size_divisibility

    def input_resize(self, images: List):
        return ImageList.from_tensors(images, self.size_divisibility)

    def input_to_features(self, x, device: str) -> Dict:
        """Computes deep features at the intermediate layer(s) all the way from the input"""

        self.model = self.model.to(device).eval()

        if self.split_id == self.SPLIT_FPN:
            return self._input_to_feature_pyramid(x)
        elif self.split_id == self.SPLIT_C2:
            return self._input_to_c2(x)
        elif self.split_id == self.SPLIT_R2:
            return self._input_to_r2(x)
        else:
            self.logger.error(f"Not supported split point {self.split_id}")

        raise NotImplementedError

    def features_to_output(self, x: Dict, device: str):
        """Complete the downstream task from the intermediate deep features"""

        self.model = self.model.to(device).eval()

        if self.split_id == self.SPLIT_FPN:
            return self._feature_pyramid_to_output(
                x["data"], x["org_input_size"], x["input_size"]
            )
        elif self.split_id == self.SPLIT_C2:
            return self._feature_c2_to_output(
                x["data"], x["org_input_size"], x["input_size"]
            )
        elif self.split_id == self.SPLIT_R2:
            return self._feature_r2_to_output(
                x["data"], x["org_input_size"], x["input_size"]
            )
        else:
            self.logger.error(f"Not supported split points {self.split_id}")

        raise NotImplementedError

    @torch.no_grad()
    def _input_to_feature_pyramid(self, x):
        """Computes and return feature pyramid ['p2', 'p3', 'p4', 'p5'] all the way from the input"""
        imgs = self.model.preprocess_image(x)
        feature_pyramid = self.backbone(imgs.tensor)
        del feature_pyramid["p6"]

        return {"data": feature_pyramid, "input_size": imgs.image_sizes}

    @torch.no_grad()
    def _input_to_c2(self, x):
        """Computes and return feature tensors at C2 from input"""
        imgs = self.model.preprocess_image(x)

        c_features = self.split_layer_list
        ref_features = self.backbone.in_features

        results = []

        # Resnet FPN
        bottom_up_features = self.backbone.bottom_up(imgs.tensor)

        for idx, lateral_conv in enumerate(self.backbone.lateral_convs):
            features = bottom_up_features[ref_features[-idx - 1]]
            results.insert(0, lateral_conv(features))

        assert len(c_features) == len(results)
        out = {f: res for f, res in zip(c_features, results)}

        return {"data": out, "input_size": imgs.image_sizes}

    @torch.no_grad()
    def _input_to_r2(self, x):
        """Computes and return feature tensor at R2 from input"""
        imgs = self.model.preprocess_image(x)

        # Resnet FPN
        stem_out = self.backbone.bottom_up.stem(imgs.tensor)
        r2_out = self.backbone.bottom_up.res2(stem_out)

        return {"data": {"r2": r2_out}, "input_size": imgs.image_sizes}

    @torch.no_grad()
    def get_input_size(self, x):
        """Computes input image size to the network"""
        imgs = self.model.preprocess_image(x)
        return imgs.image_sizes

    @torch.no_grad()
    def _feature_pyramid_to_output(
        self, x: Dict, org_img_size: Dict, input_img_size: List
    ):
        """
        performs  downstream task using the feature pyramid ['p2', 'p3', 'p4', 'p5']

        Detectron2 source codes are referenced for this function, specifically the class "GeneralizedRCNN"
        Unnecessary parts for split inference are removed or modified properly.

        Please find the license statement in the downloaded original Detectron2 source codes or at here:
        https://github.com/facebookresearch/detectron2/blob/main/LICENSE

        """

        class dummy:
            def __init__(self, img_size: list):
                self.image_sizes = img_size

        cdummy = dummy(input_img_size)

        # Replacing tag names for interfacing with NN-part2
        x = dict(zip(self.features_at_splits.keys(), x.values()))
        x.update({"p6": self.top_block(x["p5"])[0]})

        proposals, _ = self.proposal_generator(cdummy, x, None)
        results, _ = self.roi_heads(cdummy, x, proposals, None)

        assert (
            not torch.jit.is_scripting()
        ), "Scripting is not supported for postprocess."
        return self.model._postprocess(
            results,
            [
                org_img_size,
            ],
            input_img_size,
        )

    @torch.no_grad()
    def _feature_c2_to_output(self, x: Dict, org_img_size: Dict, input_img_size: List):
        """
        performs  downstream task using the c2 ['c2', 'c3', 'c4', 'c5']

        Detectron2 source codes are referenced for this function, specifically the class "GeneralizedRCNN"
        Unnecessary parts for split inference are removed or modified properly.

        Please find the license statement in the downloaded original Detectron2 source codes or at here:
        https://github.com/facebookresearch/detectron2/blob/main/LICENSE

        """
        # Replacing tag names for interfacing with NN-part2
        x = dict(zip(self.features_at_splits.keys(), x.values()))
        x = self.backbone.forward_after_c2(x)

        class dummy:
            def __init__(self, img_size: list):
                self.image_sizes = img_size

        cdummy = dummy(input_img_size)

        proposals, _ = self.proposal_generator(cdummy, x, None)
        results, _ = self.roi_heads(cdummy, x, proposals, None)

        assert (
            not torch.jit.is_scripting()
        ), "Scripting is not supported for postprocess."
        return self.model._postprocess(
            results,
            [
                org_img_size,
            ],
            input_img_size,
        )

    @torch.no_grad()
    def _feature_r2_to_output(self, x: Dict, org_img_size: Dict, input_img_size: List):
        assert "r2" in x

        r2_out = x["r2"]
        r3_out = self.backbone.bottom_up.res3(r2_out)
        r4_out = self.backbone.bottom_up.res4(r3_out)
        r5_out = self.backbone.bottom_up.res5(r4_out)

        bottom_up_features = {
            "res2": r2_out,
            "res3": r3_out,
            "res4": r4_out,
            "res5": r5_out,
        }

        fptensors = self.backbone(bottom_up_features, no_bottom_up=True)

        class dummy:
            def __init__(self, img_size: list):
                self.image_sizes = img_size

        cdummy = dummy(input_img_size)

        proposals, _ = self.proposal_generator(cdummy, fptensors, None)
        results, _ = self.roi_heads(cdummy, fptensors, proposals, None)

        assert (
            not torch.jit.is_scripting()
        ), "Scripting is not supported for postprocess."

        return self.model._postprocess(
            results,
            [
                org_img_size,
            ],
            input_img_size,
        )

    @torch.no_grad()
    def deeper_features_for_accuracy_proxy(self, x: Dict):
        """
        compute accuracy proxy at the deeper layer than NN-Part1
        """
        raise NotImplementedError

        d = {}
        for e, ft in enumerate(x["data"].values()):
            nft = ft.contiguous().to(self.device)
            assert (
                nft.dim() == 3 or nft.dim() == 4
            ), f"Input feature tensor dimension is supposed to be 3 or 4, but got {nft.dim()}"
            d[e] = nft.unsqueeze(0) if nft.dim() == 3 else nft

        class dummy:
            def __init__(self, img_size: list):
                self.image_sizes = img_size

        cdummy = dummy(x["input_size"])

        # Replacing tag names for interfacing with NN-part2
        d = dict(zip(self.features_at_splits.keys(), d.values()))
        d.update({"p6": self.top_block(d["p5"])[0]})

        proposals, _ = self.proposal_generator(cdummy, d, None)

        return proposals[0]

    @torch.no_grad()
    def forward(self, x):
        """Complete the downstream task with end-to-end manner all the way from the input"""
        # test
        return self.model([x])

    @property
    def cfg(self):
        return self._cfg


@register_vision_model("faster_rcnn_X_101_32x8d_FPN_3x")
class faster_rcnn_X_101_32x8d_FPN_3x(Rcnn_R_50_X_101_FPN):
    def __init__(self, device: str, **kwargs):
        super().__init__(device, **kwargs)


@register_vision_model("mask_rcnn_X_101_32x8d_FPN_3x")
class mask_rcnn_X_101_32x8d_FPN_3x(Rcnn_R_50_X_101_FPN):
    def __init__(self, device: str, **kwargs):
        super().__init__(device, **kwargs)


@register_vision_model("faster_rcnn_R_50_FPN_3x")
class faster_rcnn_R_50_FPN_3x(Rcnn_R_50_X_101_FPN):
    def __init__(self, device: str, **kwargs):
        super().__init__(device, **kwargs)


@register_vision_model("mask_rcnn_R_50_FPN_3x")
class mask_rcnn_R_50_FPN_3x(Rcnn_R_50_X_101_FPN):
    def __init__(self, device: str, **kwargs):
        super().__init__(device, **kwargs)
