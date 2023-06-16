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

import logging
import os
import shutil
from enum import Enum
from pathlib import Path
from typing import Callable, Dict
from uuid import uuid4 as uuid

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from compressai_vision.evaluators import BaseEvaluator
from compressai_vision.model_wrappers import BaseWrapper
from compressai_vision.registry import register_pipeline
from compressai_vision.utils import dataio

from .base_split import EXT, BaseSplit, Parts


@register_pipeline("unfold-split-inference")
class UnfoldSplitInference(BaseSplit):
    def __init__(
        self,
        configs: Dict,
        vision_model: BaseWrapper,
        codec,
        dataloader: Callable = None,
        evaluator: Callable = None,
    ):
        super().__init__(configs, vision_model, codec, dataloader, evaluator)

    def __call__(self) -> Dict:
        """Push image(s) through the encoder+decoder, returns number of bits for each image and encoded+decoded images

        Returns (nbitslist, x_hat), where nbitslist is a list of number of bits and x_hat is the image that has gone throught the encoder/decoder process
        """
        output_list = []
        for e, d in enumerate(tqdm(self.dataloader)):
            # TODO [hyomin - Make DefaultDatasetLoader compatible with Detectron2DataLoader]
            # Please reference to Detectron2 Dataset Mapper. Will face an issue when supporting Non-Detectron2-based network such as YOLO.

            org_img_size = {"height": d[0]["height"], "width": d[0]["width"]}

            cache_file = f'img_id_{d[0]["image_id"]}'

            featureT = self._from_input_to_features(d, cache_file)
            featureT["org_input_size"] = org_img_size

            res = self._compress_features(featureT, cache_file)

            dec_featureT = self._decompress_features(res["bitstream"], cache_file)

            pred = self._from_features_to_output(dec_featureT)

            self._collect_pairs(d, pred)

            out_res = d[0].copy()
            del out_res["image"], out_res["width"], out_res["height"]
            out_res["bytes"] = res["bytes"][0]
            out_res["coded_order"] = e
            out_res["org_input_size"] = (d[0]["width"], d[0]["height"])
            out_res["input_size"] = featureT["input_size"][0]
            output_list.append(out_res)

        mAP = self._evaluation()

        return {"coded_res": output_list, "mAP": mAP}

    def encode(self):
        """
        Write your own encoding behaviour including the pre-inference + compression part.

        The input is supposed to be image or video, which can be resized within this function
        before using it as input to the front part of the inference model.

        It is ideal to call this function when carrying ``encoding'' out only.
        """
        raise (AssertionError("virtual"))

    def decode(self):
        """
        Write your own decoding behaviour including the uncompression + the post-inference part.

        The input is supposed to be a bistream(s) to decode with the assigned decoder.

        It is ideal to call this function when carrying ``decoding'' out only.
        """

        raise (AssertionError("virtual"))

    def _from_input_to_features(self, x, name: str = None):
        """run the input according to a specific rquirement for input to encoder"""

        if not self._is_caching(Parts.PreInference):
            return self.vision_model.input_to_features(x)

        # Caching while processing the pre-inference computation
        _folder = self._create_folder(
            os.path.join(self._caching_folder, str(Parts.PreInference))
        )

        _caching_target = os.path.join(_folder, name + EXT)
        if self._check_cache_file(_caching_target):
            cached = torch.load(_caching_target)
        else:
            out = self.vision_model.input_to_features(x)
            torch.save(out, _caching_target)
            cached = out

        return cached

    def _from_features_to_output(self, x: Dict, name: str = None):
        """Postprocess of possibly encoded/decoded data for various tasks inlcuding for human viewing and machine analytics"""
        if not self._is_caching(Parts.PostInference):
            return self.vision_model.features_to_output(x)

        # Caching while processing the pre-inference computation
        _folder = self._create_folder(
            os.path.join(self._caching_folder, str(Parts.PostInference))
        )

        _caching_target = os.path.join(_folder, name + EXT)
        if self._check_cache_file(_caching_target):
            cached = torch.load(_caching_target)
        else:
            out = self.vision_model.features_to_output(x)
            torch.save(out, _caching_target)
            cached = out

        return cached

    def _compress_features(self, x, name: str):
        """
        Inputs: tensors of features
        Returns a list of frame bytes and a bitstream path.
        """

        if not self._is_caching(Parts.Encoder):
            return self.codec.encode(x, name)

        # Caching while processing encoding the input
        _folder = self._create_folder(
            os.path.join(self._caching_folder, str(Parts.Encoder))
        )

        _caching_target = os.path.join(_folder, name + EXT)
        if self._check_cache_file(_caching_target):
            cached = torch.load(_caching_target)
        else:
            out = self.codec.encode(x, name)
            torch.save(out, _caching_target)
            cached = out

        return cached

    def _decompress_features(self, x, name: str):
        """
        Inputs: a bitstream path
        Returns reconstructed feature tensors
        """

        if not self._is_caching(Parts.Decoder):
            return self.codec.decode(x, name)

        # Caching while processing encoding the input
        _folder = self._create_folder(
            os.path.join(self._caching_folder, str(Parts.Decoder))
        )

        _caching_target = os.path.join(_folder, name + EXT)
        if self._check_cache_file(_caching_target):
            cached = torch.load(_caching_target)
        else:
            out = self.codec.decode(x, name)
            torch.save(out, _caching_target)
            cached = out

        return cached


def stuff(args):  # This can be reference for codec parts
    # to get the current working directory
    rwYUV = dataio.readwriteYUV(device, format=dataio.PixelFormat.YUV400_10le, align=16)
    bitdepth = 10

    # packing_all_in_one = True
    # packing_all_in_one = False

    def min_max_normalization(x, minv: float, maxv: float, bitdepth=8):
        max_num_bins = (2**bitdepth) - 1

        out = ((x - minv) / (maxv - minv)).clamp_(0, 1)
        mid_level = -minv / (maxv - minv)

        return (out * max_num_bins).floor(), int(mid_level * max_num_bins + 0.5)

    def min_max_inv_normalization(x, minv: float, maxv: float, bitdepth=8):
        out = x / ((2**bitdepth) - 1)
        out = (out * (maxv - minv)) + minv
        return out

    setWriter = False
    setReader = False

    """

        features, input_img_size = model.input_to_feature_pyramid(d)

        frame, feature_size, subframe_height = model.reshape_feature_pyramid_to_frame(
            features, packing_all_in_one=packing_all_in_one
        )

        if packing_all_in_one:
            minv, maxv = test_dataset.get_min_max_across_tensors()
            normalized_frame, mid_level = min_max_normalization(
                frame, minv, maxv, bitdepth=bitdepth
            )

            ## dump yuv
            # if setWriter is False:
            #    rwYUV.setWriter("/pa/home/hyomin.choi/Projects/compressai-fcvcm/out_tensor/test.yuv", normalized_frame.size(1), normalized_frame.size(0))
            #    #setWriter = True

            # rwYUV.write_single_frame(normalized_frame, mid_level=mid_level)

            # read yuv
            # if setReader is False:
            #    rwYUV.setReader("/mnt/wekamount/RI-Users/hyomin.choi/Projects/compressai-fcvcm/out_tensor/BasketballDrill.yuv", normalized_frame.size(1), normalized_frame.size(0))
            #    rwYUV.setReader("/pa/home/hyomin.choi/Projects/compressai-fcvcm/out_tensor/test.yuv", normalized_frame.size(1), normalized_frame.size(0))
            #    setReader = True

            # loaded_normalized_frame = rwYUV.read_single_frame(e)
            # normalized_frame = rwYUV.read_single_frame(0)

            # diff = normalized_frame - loaded_normalized_frame
            # if setWriter is False:
            #    rwYUV.setWriter("/pa/home/hyomin.choi/Projects/compressai-fcvcm/out_tensor/diff.yuv", normalized_frame.size(1), normalized_frame.size(0))
            #    setWriter = True

            # rwYUV.write_single_frame((diff+256), mid_level=mid_level)

            rescaled_frame = min_max_inv_normalization(
                normalized_frame, minv, maxv, bitdepth=bitdepth
            )
        else:
            rescaled_frame = frame

        back_to_features = model.reshape_frame_to_feature_pyramid(
            rescaled_frame,
            feature_size,
            subframe_height,
            packing_all_in_one=packing_all_in_one,
        )

        # results = model(d)
        # print(type(results))

        evaluator.process(d, results)

    """
