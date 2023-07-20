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

from pathlib import Path
from typing import Dict, List

import torch
from jde.models import Darknet
from jde.utils.kalman_filter import KalmanFilter
from torch import Tensor

from compressai_vision.model_wrappers.utils import compute_frame_resolution
from compressai_vision.registry import register_vision_model

from .base_wrapper import BaseWrapper
from .utils import tensor_to_tiled, tiled_to_tensor

__all__ = [
    "jde_1088x608",
]

thisdir = Path(__file__).parent
root_path = thisdir.joinpath("../..")


@register_vision_model("jde_1088x608")
class jde_1088x608(BaseWrapper):
    r"""Re-implementation of JDE from Z. Wang, L. Zheng, Y. Liu, and S. Wang:
    : `"Towards Real-Time Multi-Object Tracking"`_,
    The European Conference on Computer Vision (ECCV), 2020

    The implementation refers to
    <https://github.com/Zhongdao/Towards-Realtime-MOT/blob/master/tracker/multitracker.py>

    Full license statement can be found at
    <https://github.com/Zhongdao/Towards-Realtime-MOT/blob/master/LICENSE>

    """

    def __init__(self, device="cpu", **kwargs):
        super().__init__()

        self.device = device
        self.model_info = {
            "cfg": f"{root_path}/{kwargs['cfg']}",
            "weight": f"{root_path}/{kwargs['weight']}",
            "iou_thres": float(kwargs["iou_thres"]),
            "conf_thres": float(kwargs["conf_thres"]),
            "nms_thres": float(kwargs["nms_thres"]),
            "min_box_area": int(kwargs["min_box_area"]),
            "track_buffer": int(kwargs["track_buffer"]),
        }

        assert "splits" in kwargs, "Split layer ids must be provided"
        layer_list = kwargs["splits"]
        self.features_at_splits = dict(zip(layer_list, [None] * len(layer_list)))

        self.darknet = Darknet(self.model_info["cfg"], device, nID=14455)
        self.darknet.load_state_dict(
            torch.load(self.model_info["weight"], map_location="cpu")["model"],
            strict=False,
        )
        self.darknet.to(device).eval()

        self.kalman_filter = KalmanFilter()

        # ? self.tracked_stracks = []
        # ? self.lost_stracks = []
        # ? self.removed_stracks = []

        # ? self.frame_id = 0
        # ? self.det_thresh = self.model_info['conf_thres']
        # ? self.buffer_size = int(frame_rate / 30.0 * self.model_info['track_buffer'])
        # ? self.max_time_lost = self.buffer_size

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
        """Computes and return feture pyramid all the way from the input"""
        img = x[0]["image"].unsqueeze(0).to(self.device)
        input_size = tuple(img.shape[2:])
        _ = self.darknet(img, self.features_at_splits, is_nn_part1=True)
        return {"data": self.features_at_splits, "input_size": [input_size]}

    @torch.no_grad()
    def _feature_pyramid_to_output(
        self, x: Dict, org_img_size: Dict, input_img_size: List
    ):
        """
        performs downstream task using the feature pyramid
        """
        pred = self.darknet(None, x, is_nn_part1=False)

        return pred

    @torch.no_grad()
    def forward(self, x):
        """Complete the downstream task with end-to-end manner all the way from the input"""
        return self.model(x)

    def reshape_feature_pyramid_to_frame(self, x: Dict, packing_all_in_one=False):
        """rehape the feature pyramid to a frame"""

        # 'p2' is the base for the size of to-be-formed frame

        _, C, H, W = x["p2"].size()
        _, fixedW = compute_frame_resolution(C, H, W)

        tiled_frame = {}
        feature_size = {}
        subframe_heights = {}
        for key, tensor in x.items():
            N, C, H, W = tensor.size()

            assert N == 1, f"the batch size shall be one, but got {N}"

            frmH, frmW = compute_frame_resolution(C, H, W)

            rescale = fixedW // frmW if packing_all_in_one else 1

            new_frmH = frmH // rescale
            new_frmW = frmW * rescale

            frame = tensor_to_tiled(tensor, (new_frmH, new_frmW))

            tiled_frame.update({key: frame})
            feature_size.update({key: tensor.size()})
            subframe_heights.update({key: new_frmH})

        if packing_all_in_one:
            for key, subframe in tiled_frame.items():
                if key == "p2":
                    out = subframe
                else:
                    out = torch.cat([out, subframe], dim=0)
            tiled_frame = out

        return tiled_frame, feature_size, subframe_heights

    def reshape_frame_to_feature_pyramid(
        self, x, tensor_shape: Dict, subframe_height: Dict, packing_all_in_one=False
    ):
        """reshape a frame of channels into the feature pyramid"""

        assert isinstance(x, (Tensor, Dict))

        top_y = 0
        tiled_frame = {}
        if packing_all_in_one:
            for key, height in subframe_height.items():
                tiled_frame.update({key: x[top_y : top_y + height, :]})
                top_y = top_y + height
        else:
            assert isinstance(x, Dict)
            tiled_frame = x

        feature_tensor = {}
        for key, frame in tiled_frame.items():
            _, numChs, chH, chW = tensor_shape[key]
            tensor = tiled_to_tensor(frame, (chH, chW)).to(self.device)
            assert tensor.size(1) == numChs

            feature_tensor.update({key: tensor})

        return feature_tensor

    # [TODO (choih): To be reused for some purpose]
    def preInputTensor(self, img, img_id):
        """

        :param img: numpy BGR image (h,w,3)

        """
        height, width = img.shape[:2]
        if self.aug is not None:
            image = self.aug.get_transform(img).apply_image(img)
        image = torch.as_tensor(image.astype("float32").transpose(2, 0, 1))

        inputs = {
            "image": image,
            "height": height,
            "width": width,
            "image_id": img_id,
        }
        return [
            inputs,
        ]
