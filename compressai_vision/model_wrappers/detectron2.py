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

import os
from pathlib import Path
from typing import Any, Dict, List

import torch
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.modeling import build_model
from torch import Tensor

from compressai_vision.model_wrappers.utils import compute_frame_resolution
from compressai_vision.registry import register_vision_model

from .base_wrapper import BaseWrapper
from .utils import tensor_to_tiled, tiled_to_tensor

__all__ = [
    "faster_rcnn_X_101_32x8d_FPN_3x",
    "mask_rcnn_X_101_32x8d_FPN_3x",
    "faster_rcnn_R_50_FPN_3x",
    "mask_rcnn_R_50_FPN_3x",
]

thisdir = Path(__file__).parent
root_path = thisdir.joinpath("../..")


class Rcnn_R_50_X_101_FPN(BaseWrapper):
    def __init__(self, device="cpu", **kwargs):
        super().__init__()

        self.device = device
        self._cfg = get_cfg()
        self._cfg.MODEL.DEVICE = device
        self._cfg.merge_from_file(f"{root_path}/{kwargs['cfg']}")
        self.model = build_model(self._cfg).to(device).eval()

        self.backbone = self.model.backbone
        self.top_block = self.model.backbone.top_block
        self.proposal_generator = self.model.proposal_generator
        self.roi_heads = self.model.roi_heads
        self.postprocess = self.model._postprocess
        DetectionCheckpointer(self.model).load(f"{root_path}/{kwargs['weight']}")

        self.model_info = {"cfg": kwargs["cfg"], "weight": kwargs["weight"]}

        assert "splits" in kwargs, "Split layer ids must be provided"
        self.split_layer_list = kwargs["splits"]
        self.features_at_splits = dict(
            zip(self.split_layer_list, [None] * len(self.split_layer_list))
        )

        assert self.top_block is not None
        assert self.proposal_generator is not None

    def input_to_features(self, x) -> Dict:
        """Computes deep features at the intermediate layer(s) all the way from the input"""
        return self._input_to_feature_pyramid(x)

    def features_to_output(self, x: Dict):
        """Complete the downstream task from the intermediate deep features"""
        return self._feature_pyramid_to_output(
            x["data"], x["org_input_size"], x["input_size"]
        )

    @torch.no_grad()
    def _input_to_feature_pyramid(self, x):
        """Computes and return feture pyramid ['p2', 'p3', 'p4', 'p5'] all the way from the input"""
        imgs = self.model.preprocess_image(x)
        feature_pyramid = self.backbone(imgs.tensor)
        del feature_pyramid["p6"]

        return {"data": feature_pyramid, "input_size": imgs.image_sizes}

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
    def deeper_features_for_accuracy_proxy(self, x: Dict):
        """
        compute accuracy proxy at the deeper layer than NN-Part1
        """

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
        return self.model(x)

    def reshape_feature_pyramid_to_frame(self, x: Dict, packing_all_in_one=False):
        """rehape the feature pyramid to a frame"""

        # 'p2' is the base for the size of to-be-formed frame

        nbframes, C, H, W = x["p2"].size()
        _, fixedW = compute_frame_resolution(C, H, W)

        packed_frames = {}
        feature_size = {}
        subframe_heights = {}
        subframe_widths = {}

        assert packing_all_in_one == True, "False is not support yet"

        packed_frame_list = []
        for n in range(nbframes):
            for key, tensor in x.items():
                single_tensor = tensor[n : n + 1, ::]
                N, C, H, W = single_tensor.size()

                assert N == 1, f"the batch size shall be one, but got {N}"

                if n == 0:
                    feature_size.update({key: single_tensor.size()})

                    frmH, frmW = compute_frame_resolution(C, H, W)

                    rescale = fixedW // frmW if packing_all_in_one else 1

                    new_frmH = frmH // rescale
                    new_frmW = frmW * rescale

                    subframe_heights.update({key: new_frmH})
                    subframe_widths.update({key: new_frmW})

                tile = tensor_to_tiled(
                    single_tensor, (subframe_heights[key], subframe_widths[key])
                )

                packed_frames.update({key: tile})

            if packing_all_in_one:
                for key, subframe in packed_frames.items():
                    if key == "p2":
                        out = subframe
                    else:
                        out = torch.cat([out, subframe], dim=0)

                packed_frame_list.append(out)

        packed_frames = torch.stack(packed_frame_list)

        return packed_frames, feature_size, subframe_heights

    def reshape_frame_to_feature_pyramid(
        self, x, tensor_shape: Dict, subframe_height: Dict, packing_all_in_one=False
    ):
        """reshape a frame of channels into the feature pyramid"""

        assert isinstance(x, (Tensor, Dict))
        assert packing_all_in_one is True, "False is not supported yet"

        top_y = 0
        tiled_frames = {}
        if packing_all_in_one:
            for key, height in subframe_height.items():
                tiled_frames.update({key: x[:, top_y : top_y + height, :]})
                top_y = top_y + height
        else:
            raise NotImplementedError
            assert isinstance(x, Dict)
            tiled_frames = x

        feature_tensor = {}
        for key, frames in tiled_frames.items():
            _, numChs, chH, chW = tensor_shape[key]

            tensors = []
            for frame in frames:
                tensor = tiled_to_tensor(frame, (chH, chW)).to(self.device)
                tensors.append(tensor)
            tensors = torch.cat(tensors, dim=0)
            assert tensors.size(1) == numChs

            feature_tensor.update({key: tensors})

        return feature_tensor

    @property
    def cfg(self):
        return self._cfg


@register_vision_model("faster_rcnn_X_101_32x8d_FPN_3x")
class faster_rcnn_X_101_32x8d_FPN_3x(Rcnn_R_50_X_101_FPN):
    def __init__(self, device="cpu", **kwargs):
        super().__init__(device, **kwargs)


@register_vision_model("mask_rcnn_X_101_32x8d_FPN_3x")
class mask_rcnn_X_101_32x8d_FPN_3x(Rcnn_R_50_X_101_FPN):
    def __init__(self, device="cpu", **kwargs):
        super().__init__(device, **kwargs)


@register_vision_model("faster_rcnn_R_50_FPN_3x")
class faster_rcnn_R_50_FPN_3x(Rcnn_R_50_X_101_FPN):
    def __init__(self, device="cpu", **kwargs):
        super().__init__(device, **kwargs)


@register_vision_model("mask_rcnn_R_50_FPN_3x")
class mask_rcnn_R_50_FPN_3x(Rcnn_R_50_X_101_FPN):
    def __init__(self, device="cpu", **kwargs):
        super().__init__(device, **kwargs)
