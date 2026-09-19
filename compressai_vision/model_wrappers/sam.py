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

# sam = sam_model_registry["vit_h"](checkpoint="/t/vic/hevc_simulations/rosen/build/compressai13_sam/weights/sam/sam_vit_h_4b8939.pth")
# predictor = SamPredictor(sam)
# mask_generator = SamAutomaticMaskGenerator(sam, points_per_batch=16)


__all__ = [
    "sam_vit_h_4b8939",
    "sam_vit_b_01ec64",
    "sam_vit_l_0b3195",
]

thisdir = Path(__file__).parent
root_path = thisdir.joinpath("../..")


def mask_to_bbx(mask):
    mask = mask.cpu()
    mask = np.squeeze(np.array(mask))
    h, w = mask.shape[-2:]
    rows, cols = np.where(mask)

    # Calculate the bounding box
    # min_row, max_row = rows.min(), rows.max()
    # min_col, max_col = cols.min(), cols.max()

    # Bounding box as (top-left corner, bottom-right corner)
    bounding_box = [
        cols.min(),
        rows.min(),
        cols.max(),
        rows.max(),
    ]  # XMin,YMin,XMax,YMax  top-left corner, bottom-right corner
    # bounding_box = cols.min()/w, cols.max()/w, rows.min()/h, rows.max()/h  # XMin,XMax,YMin,YMax
    return bounding_box


class Split_Points(Enum):
    def __str__(self):
        return str(self.value)

    ImageEncoder = "imgenc"  # features output from neck.3.bias
    GlobalAttn2 = "global_attn2"


class SAM(BaseWrapper):
    def __init__(self, device: str, **kwargs):
        from segment_anything import sam_model_registry
        from segment_anything.utils.transforms import ResizeLongestSide

        super().__init__(device)
        assert kwargs.get(
            "prompts_folder"
        ), "dataset.config.prompts_folder must be provided for SAM"
        self.prompts_folder = Path(kwargs["prompts_folder"])
        assert kwargs.get("prompt_format") in {
            "points_class",
            "point_label_class",
        }, "Unsupported dataset.config.prompt_format for SAM"
        self.prompt_format = kwargs["prompt_format"]
        self.sam_model_registry = sam_model_registry

        _path_prefix = (
            f"{root_path}"
            if kwargs["model_path_prefix"] == "default"
            else kwargs["model_path_prefix"]
        )
        self.model_info = {
            "cfg": f"{_path_prefix}/{kwargs['cfg']}",
            "weights": f"{_path_prefix}/{kwargs['weights']}",
        }

        self.model = (
            self.sam_model_registry["vit_h"](checkpoint=self.model_info["weights"])
            .to(device)
            .eval()
        )

        self.image_encoder = self.model.image_encoder
        self.prompt_encoder = self.model.prompt_encoder
        self.head = self.model.mask_decoder

        self.supported_split_points = Split_Points

        assert "splits" in kwargs, "Split layer ids must be provided"
        self.split_id = str(kwargs["splits"]).lower()

        if self.split_id == str(self.supported_split_points.ImageEncoder):
            self.split_layer_list = [
                "imgenc",
            ]
        elif self.split_id == str(self.supported_split_points.GlobalAttn2):
            self.split_layer_list = [
                "global_attn2",
            ]
        else:
            raise NotImplementedError

        self.features_at_splits = dict(
            zip(self.split_layer_list, [None] * len(self.split_layer_list))
        )

        self.sam_resize = ResizeLongestSide(self.image_encoder.img_size)

    @property
    def SPLIT_IMGENC(self):
        return str(self.supported_split_points.ImageEncoder)

    @property
    def SPLIT_GLOBAL_ATTN2(self):
        return str(self.supported_split_points.GlobalAttn2)

    def prompt_inputs(self, file_name):
        # [TODO] should be improved...
        prompt_link = self.prompts_folder / f"{Path(file_name).stem}.txt"

        prompts = []
        prompt_labels = []
        object_classes = []

        with open(prompt_link, "r") as f:
            for line in f:
                parts = list(map(int, line.strip().split()))
                object_class = parts[-1] if parts else None

                if self.prompt_format == "points_class":
                    valid = len(parts) >= 3 and len(parts) % 2 == 1
                    step = 2
                else:
                    valid = len(parts) >= 4 and (len(parts) - 1) % 3 == 0
                    step = 3

                if not valid:
                    raise ValueError(
                        f"Invalid prompt format in {prompt_link}: {line.strip()}"
                    )

                coord_values = parts[:-1]
                points = [
                    [coord_values[i], coord_values[i + 1]]
                    for i in range(0, len(coord_values), step)
                ]
                labels = (
                    [1] * len(points)
                    if self.prompt_format == "points_class"
                    else [
                        coord_values[i + 2] for i in range(0, len(coord_values), step)
                    ]
                )

                prompts.append(points)
                prompt_labels.append(labels)
                object_classes.append(object_class)

        return prompts, prompt_labels, object_classes

    def input_to_features(self, x: list, device: str) -> Dict:
        """Computes deep features at the intermediate layer(s) all the way from the input"""
        self.model = self.model.to(device).eval()
        assert isinstance(x, list) and len(x) == 1

        if self.split_id == self.SPLIT_IMGENC:
            return self._input_to_image_encoder(x, device)
        elif self.split_id == self.SPLIT_GLOBAL_ATTN2:
            return self._input_to_global_attn2(x, device)
        else:
            self.logger.error(f"Not supported split point {self.split_id}")

        raise NotImplementedError

    def features_to_output(self, x: Dict, device: str):
        """Complete the downstream task from the intermediate deep features"""

        self.model = self.model.to(device).eval()

        if self.split_id == self.SPLIT_IMGENC:
            assert "file_name" in x

            prompts, prompt_labels, object_classes = self.prompt_inputs(x["file_name"])

            return self._image_encoder_to_output(
                x["data"],
                x["org_input_size"],
                x["input_size"],
                prompts,
                prompt_labels,
                object_classes,
                device,
            )
        elif self.split_id == self.SPLIT_GLOBAL_ATTN2:
            assert "file_name" in x

            prompts, prompt_labels, object_classes = self.prompt_inputs(x["file_name"])

            return self._global_attn2_to_output(
                x["data"],
                x["org_input_size"],
                x["input_size"],
                prompts,
                prompt_labels,
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

        img = x[0]["image"].to(device)
        input_size = list(img.size()[2:])
        feature = {}
        input_img = self.model.preprocess(img)
        feature["backbone"] = self.image_encoder(input_img)

        return {
            "data": feature,
            "input_size": input_size,
        }

    @torch.no_grad()
    def _input_to_global_attn2(self, x, device):
        """Computes and return encoded image all the way from the input"""
        assert len(x) == 1

        img = x[0]["image"].to(device)
        input_size = list(img.size()[2:])
        feature = {}
        input_img = self.model.preprocess(img)
        feature_ = self.image_encoder.patch_embed(input_img)
        if self.image_encoder.pos_embed is not None:
            feature_ = feature_ + self.image_encoder.pos_embed

        split_block_idx = self._global_attn2_block_index()
        for block in self.image_encoder.blocks[: split_block_idx + 1]:
            feature_ = block(feature_)

        feature_ = feature_.permute(0, 3, 1, 2)  # intermediate features to output

        feature["backbone"] = feature_
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
        prompt_labels: List,
        object_classes: List,
        device,
    ):
        """Performs the downstream task using the encoded image feature."""
        return self._image_embeddings_to_output(
            image_embeddings=x["imgenc"],
            org_img_size=org_img_size,
            input_img_size=input_img_size,
            prompts=prompts,
            prompt_labels=prompt_labels,
            object_classes=object_classes,
            device=device,
        )

    @torch.no_grad()
    def _global_attn2_to_output(
        self,
        x: Dict,
        org_img_size: Dict,
        input_img_size: List,
        prompts: List,
        prompt_labels: List,
        object_classes: List,
        device,
    ):
        """Performs the downstream task from the global_attn2 features."""
        image_embeddings = self._global_attn2_to_image_embeddings(x["global_attn2"])

        return self._image_embeddings_to_output(
            image_embeddings=image_embeddings,
            org_img_size=org_img_size,
            input_img_size=input_img_size,
            prompts=prompts,
            prompt_labels=prompt_labels,
            object_classes=object_classes,
            device=device,
        )

    def _global_attn2_to_image_embeddings(
        self,
        global_attn2_features: torch.Tensor,
    ) -> torch.Tensor:
        """Runs the remaining image-encoder layers after global_attn2."""
        features = global_attn2_features.permute(0, 2, 3, 1)

        split_block_idx = self._global_attn2_block_index()
        for block in self.image_encoder.blocks[split_block_idx + 1 :]:
            features = block(features)

        return self.image_encoder.neck(features.permute(0, 3, 1, 2))

    def _global_attn2_block_index(self) -> int:
        """Returns the image-encoder block index for the second global attention."""
        global_attn_indexes = [
            index
            for index, block in enumerate(self.image_encoder.blocks)
            if block.window_size == 0
        ]
        if len(global_attn_indexes) < 2:
            raise RuntimeError(
                "The SAM image encoder does not contain a second global-attention block"
            )

        return global_attn_indexes[1]

    def _image_embeddings_to_output(
        self,
        image_embeddings: torch.Tensor,
        org_img_size: Dict,
        input_img_size: List,
        prompts: List,
        prompt_labels: List,
        object_classes: List,
        device,
    ):
        """Runs prompt encoding, mask decoding, and output postprocessing."""
        from detectron2.structures import Boxes, Instances

        instances_list = []

        for prompt, labels, object_class in zip(
            prompts,
            prompt_labels,
            object_classes,
        ):
            input_points = np.asarray(prompt, dtype=np.float32)
            input_points = self.sam_resize.apply_coords(
                input_points,
                (org_img_size["height"], org_img_size["width"]),
            )
            input_points = torch.as_tensor(
                input_points,
                dtype=torch.float32,
                device=device,
            ).unsqueeze(0)

            input_labels = torch.as_tensor(
                labels,
                dtype=torch.int64,
                device=device,
            ).unsqueeze(0)
            points = (input_points, input_labels)

            sparse_embeddings, dense_embeddings = self.prompt_encoder(
                points=points,
                boxes=None,
                masks=None,
            )
            image_pe = self.prompt_encoder.get_dense_pe()

            low_res_masks, iou_pred = self.model.mask_decoder(
                image_embeddings=image_embeddings,
                image_pe=image_pe,
                sparse_prompt_embeddings=sparse_embeddings,
                dense_prompt_embeddings=dense_embeddings,
                multimask_output=False,
            )

            masks = F.interpolate(
                low_res_masks,
                (self.image_encoder.img_size, self.image_encoder.img_size),
                mode="bilinear",
                align_corners=False,
            )
            masks = masks[..., : input_img_size[0], : input_img_size[1]]
            masks = F.interpolate(
                masks,
                (org_img_size["height"], org_img_size["width"]),
                mode="bilinear",
                align_corners=False,
            )
            masks = masks > 0.0

            boxes = mask_to_bbx(masks[0])
            boxes = Boxes(torch.tensor(np.array([boxes])))
            scores = torch.tensor([iou_pred])
            classes = torch.tensor([object_class], dtype=torch.int64)

            instances = Instances(
                image_size=(org_img_size["height"], org_img_size["width"])
            )
            instances.set("pred_boxes", boxes)
            instances.set("scores", scores)
            instances.set("pred_classes", classes)
            instances.set("pred_masks", masks[0])
            instances_list.append(instances)

        return [{"instances": Instances.cat(instances_list)}]

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

        prompts, prompt_labels, object_classes = self.prompt_inputs(x["file_origin"])

        dec_res = self._image_encoder_to_output(
            enc_res["data"],
            {"height": x["height"], "width": x["width"]},
            enc_res["input_size"],
            prompts,
            prompt_labels,
            object_classes,
            device=self.device,
        )

        return dec_res


@register_vision_model("sam_vit_h_4b8939")
class sam_vit_h_4b8939(SAM):
    def __init__(self, device: str, **kwargs):
        super().__init__(device, **kwargs)


@register_vision_model("sam_vit_b_01ec64")
class sam_vit_b_01ec64(SAM):
    def __init__(self, device: str, **kwargs):
        super().__init__(device, **kwargs)


@register_vision_model("sam_vit_l_0b3195")
class sam_vit_l_0b3195(SAM):
    def __init__(self, device: str, **kwargs):
        super().__init__(device, **kwargs)
