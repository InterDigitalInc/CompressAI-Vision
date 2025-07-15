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
from detectron2.structures import ImageList, Instances
from segment_anything import (  # , Instances
    SamAutomaticMaskGenerator,
    SamPredictor,
    sam_model_registry,
)
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


class Boxes:
    def __init__(self, tensor):
        self.tensor = tensor

    def __len__(self):
        return self.tensor.shape[0]

    def __repr__(self):
        return f"Boxes(tensor({self.tensor}))"


def mask_to_bbx(mask):
    mask = np.array(mask)
    mask = np.squeeze(mask)
    h, w = mask.shape[-2:]
    rows, cols = np.where(mask)

    # Calculate the bounding box
    min_row, max_row = rows.min(), rows.max()
    min_col, max_col = cols.min(), cols.max()

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


class SAM(BaseWrapper):
    def __init__(self, device: str, **kwargs):
        super().__init__(device)

        self.model = (
            sam_model_registry["vit_h"](checkpoint=kwargs["weights"]).to(device).eval()
        )
        self.model.load_state_dict(torch.load(kwargs["weights"]))

        self.backbone = self.model.image_encoder
        self.prompt_encoder = self.model.prompt_encoder
        self.head = self.model.mask_decoder

        # SamPredictor(self.model)
        # print(SamPredictor)
        self.supported_split_points = Split_Points

        assert "splits" in kwargs, "Split layer ids must be provided"
        self.split_id = str(kwargs["splits"]).lower()

        # if self.split_id == str(self.supported_split_points.ImageEncoder):
        self.split_layer_list = ["imgenc"]
        # else:
        #    raise NotImplementedError

        self.features_at_splits = dict(
            zip(self.split_layer_list, [None] * len(self.split_layer_list))
        )

        self.annotation_file = "/o/projects/proj-river/ctc_sequences/vcm_testdata/samtest/annotations/mpeg-oiv6-segmentation-coco_fortest.json"

    @property
    def SPLIT_IMGENC(self):
        return str(self.supported_split_points.ImageEncoder)

    def input_to_features(self, x, device: str) -> Dict:
        """Computes deep features at the intermediate layer(s) all the way from the input"""
        self.model = self.model.to(device).eval()

        if self.split_id == self.SPLIT_IMGENC:
            return self._input_to_image_encoder(x)
        else:
            self.logger.error(f"Not supported split point {self.split_id}")

        raise NotImplementedError

    def features_to_output(self, x: Dict, device: str):
        """Complete the downstream task from the intermediate deep features"""

        self.model = self.model.to(device).eval()

        if self.split_id == self.SPLIT_IMGENC:
            return self._image_encoder_to_output(
                x["data"],
                x["org_input_size"],
                x["input_size"],
                x["prompts"],
                x["object_classes"],
            )
        else:
            self.logger.error(f"Not supported split points {self.split_id}")

        raise NotImplementedError

    @torch.no_grad()
    def _input_to_image_encoder(self, x):
        """Computes and return encoded image all the way from the input"""
        # TODO pre_processing
        # print("AAAAA _input_to_image_encoder", x ,'\n')
        # imgs = ImageList(x)
        imgs = x[0]["image"]
        feature = {}
        feature["backbone"] = self.backbone(imgs)

        prompt_link = (
            x[0]["file_name"].replace("/images/", "/prompts/").replace(".jpg", ".txt")
        )
        # print("AAAAA prompt_link", prompt_link)

        with open(prompt_link, "r") as f:
            line = f.readline()
            # first_two = list(map(int, line.strip().split()[:2]))
            parts = line.strip().split()
            prompts = list(map(int, parts[:2]))
            object_classes = [int(line.strip().split()[-1])]

        image_sizes = [x[0]["height"], x[0]["width"]]
        # print("AAAAA image_sizes", image_sizes, int(image_sizes[0]) * int(image_sizes[1])),

        return {
            "data": feature,
            "input_size": image_sizes,
            "prompts": prompts,
            "object_classes": object_classes,
        }

    @torch.no_grad()
    def get_input_size(self, x):
        """Computes input image size to the network"""
        # TODO

        image_sizes = [x[0]["height"], x[0]["width"]]
        return image_sizes  # [1024, 1024]

    @torch.no_grad()
    def get_prompts(self, x):
        """Computes prompts"""
        prompt_link = (
            x[0]["file_name"].replace("/images/", "/prompts/").replace(".jpg", ".txt")
        )
        # print("AAAAA prompt_link", prompt_link)

        with open(prompt_link, "r") as f:
            line = f.readline()
            # first_two = list(map(int, line.strip().split()[:2]))
            parts = line.strip().split()
            prompts = list(map(int, parts[:2]))
            object_classes = [int(line.strip().split()[-1])]

        image_sizes = [x[0]["height"], x[0]["width"]]
        # print("AAAAA image_sizes", image_sizes, int(image_sizes[0]) * int(image_sizes[1])),

        return prompts

    @torch.no_grad()
    def get_object_classes(self, x):
        """Computes input image size to the network"""
        prompt_link = (
            x[0]["file_name"].replace("/images/", "/prompts/").replace(".jpg", ".txt")
        )
        # print("AAAAA prompt_link", prompt_link)

        with open(prompt_link, "r") as f:
            line = f.readline()
            # first_two = list(map(int, line.strip().split()[:2]))
            parts = line.strip().split()
            prompts = list(map(int, parts[:2]))
            object_classes = [int(line.strip().split()[-1])]

        image_sizes = [x[0]["height"], x[0]["width"]]
        # print("AAAAA image_sizes", image_sizes, int(image_sizes[0]) * int(image_sizes[1])),
        return object_classes

    @torch.no_grad()
    def _image_encoder_to_output(
        self,
        x: Dict,
        org_img_size: Dict,
        input_img_size: List,
        prompts: List,
        object_classes: List,
    ):
        """
        performs  downstream task using the encoded image feature

        """
        # print("prompts object_classes", prompts,  object_classes)

        input_points = [prompts]  # [[469, 295]] #prompts["points"]
        input_points = np.array(input_points)
        input_points_ = torch.tensor(input_points)
        input_points_ = input_points_.unsqueeze(-1)
        input_points_ = input_points_.permute(2, 0, 1)

        input_labels = np.array([1])
        input_labels_ = torch.tensor(input_labels)
        input_labels_ = input_labels_.unsqueeze(-1)
        input_labels_ = input_labels_.permute(1, 0)

        points = (torch.tensor(input_points_), torch.tensor(input_labels_))
        prompt_feature = self.prompt_encoder(points=points, boxes=None, masks=None)
        image_pe = self.prompt_encoder.get_dense_pe()

        low_res_masks, iou_pred = self.model.mask_decoder(
            image_embeddings=x["imgenc"],
            image_pe=image_pe,
            sparse_prompt_embeddings=prompt_feature[0],
            dense_prompt_embeddings=prompt_feature[1],
            multimask_output=False,
        )
        # print("len low_res_masks", len(low_res_masks))
        # post process mask
        masks = F.interpolate(
            low_res_masks,
            (1024, 1024),
            mode="bilinear",
            align_corners=False,
        )
        masks = masks[
            ..., : input_img_size[0], : input_img_size[1]
        ]  # [..., : 793, : 1024]
        masks = F.interpolate(
            masks,
            (input_img_size[0], input_img_size[1]),
            mode="bilinear",
            align_corners=False,
        )

        # masks1 = self.postprocess_masks(
        #        masks= low_res_masks,
        #        input_size=input_img_size,
        #        original_size=org_img_size,
        #    )
        mask_threshold = 0.0
        masks = masks > mask_threshold
        # print("len masks", len(masks), masks[0].shape)
        # name = '/t/vic/hevc_simulations/rosen/build/main-20250423-sam1/masks' + str(input_img_size[0]) + '.pt'
        # torch.save(masks[0], name) #"/t/vic/hevc_simulations/rosen/build/main-20250423-sam1/masks.pt")

        # post process result
        processed_results = []
        boxes = mask_to_bbx(masks[0])
        # print("boxes", boxes)
        boxes = Boxes(torch.tensor(np.array([boxes])))
        scores = torch.tensor([iou_pred])
        classes = torch.tensor(object_classes)  # 48 for sandwich,
        # masks = torch.rand(1, 683, 1024)  # Example binary mask

        # Create an instance
        instances = Instances(image_size=(input_img_size[0], input_img_size[1]))
        instances.set("pred_boxes", boxes)
        instances.set("scores", scores)
        instances.set("pred_classes", classes)
        instances.set("pred_masks", masks[0])  # ✅ Now a real tensor

        # Wrap in result
        # result = [f"{{'instances': {instances}}}"]
        # print("result", result)
        # print("instances", instances.get_fields().keys(), len(instances))
        processed_results.append({"instances": instances})
        # print("processed_results", len(processed_results))
        return processed_results

    @torch.no_grad()
    def forward(self, x):
        """Complete the downstream task with end-to-end manner all the way from the input"""
        # test
        enc = self._input_to_image_encoder(self, x)
        dec = self._image_encoder_to_output(enc)

        return dec

    # @property
    # def cfg(self):
    #    return self._cfg


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
