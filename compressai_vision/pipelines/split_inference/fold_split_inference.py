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


from typing import Dict

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from compressai_vision.evaluators import BaseEvaluator
from compressai_vision.model_wrappers import BaseWrapper
from compressai_vision.registry import register_pipeline
from compressai_vision.utils import dataio, to_cpu

from .base_split import BaseSplit


@register_pipeline("fold-split-inference")
class FoldSplitInference(BaseSplit):
    def __init__(
        self,
        configs: Dict,
        device: str,
    ):
        super().__init__(configs, device)

        self._fold_data_buffer = None

    def __call__(
        self,
        vision_model: BaseWrapper,
        codec,
        dataloader: DataLoader,
        evaluator: BaseEvaluator,
    ) -> Dict:
        """Push image(s) through the encoder+decoder, returns number of bits for each image and encoded+decoded images

        Returns (nbitslist, x_hat), where nbitslist is a list of number of bits and x_hat is the image that has gone throught the encoder/decoder process
        """

        featureT = {}
        gt_inputs = []
        for e, d in enumerate(tqdm(dataloader)):
            # TODO [hyomin - Make DefaultDatasetLoader compatible with Detectron2DataLoader]
            # Please reference to Detectron2 Dataset Mapper. Will face an issue when supporting Non-Detectron2-based network such as YOLO.

            cache_file = f'img_id_{d[0]["image_id"]}'
            gt_inputs.append(
                [
                    {"image_id": d[0]["image_id"]},
                ]
            )

            res = self._from_input_to_features(vision_model, d, cache_file)

            assert "data" in res
            num_items = self._data_buffering(res["data"])

            if e == 0:
                org_img_size = {"height": d[0]["height"], "width": d[0]["width"]}
                featureT["org_input_size"] = org_img_size
                assert "input_size" in res
                featureT["input_size"] = res["input_size"]

                out_res = d[0].copy()
                del out_res["image"], out_res["width"], out_res["height"]
                out_res["org_input_size"] = (d[0]["width"], d[0]["height"])
                out_res["input_size"] = featureT["input_size"][0]

        assert num_items == len(dataloader)

        # concatenate a list of tensors at each keyword item
        featureT["data"] = self._concat_data()

        res = self._compress_features(codec, featureT, cache_file)

        dec_featureT = self._decompress_features(codec, res["bitstream"], cache_file)

        # separate a tensor of each keyword item into a list of tensors
        dec_feature_tesnor = self._split_data(dec_featureT["data"])

        for e, data in enumerate(self._iterate_items(dec_feature_tesnor, num_items)):
            dec_featureT["data"] = data
            pred = self._from_features_to_output(vision_model, dec_featureT)
            evaluator.digest(gt_inputs[e], pred)

        output_list = []
        out_res["bytes"] = res["bytes"][0]
        out_res["coded_order"] = e

        output_list.append(out_res)

        mAP = self._evaluation(evaluator)

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

    def _data_buffering(self, data: Dict):
        """
        Piling up input data along the 0 axis for each dictionary item
        """

        if self._fold_data_buffer is None:
            self._fold_data_buffer = {}

            for key, tensor in data.items():
                self._fold_data_buffer[key] = [
                    to_cpu(tensor),
                ]
            return

        for key, tensor in data.items():
            self._fold_data_buffer[key].append(to_cpu(tensor))

        len_items = [len(buffer) for buffer in self._fold_data_buffer.values()]

        return len_items[0]

    def _concat_data(self):
        output = {}
        for key, tensor in self._fold_data_buffer.items():
            output[key] = torch.concat(tensor, dim=0)

        self._fold_data_buffer = None

        return output

    def _split_data(self, data: Dict):
        output = {}
        for key, tensor in data.items():
            output[key] = list(tensor.chunk(len(tensor)))
        return output

    def _iterate_items(self, data: Dict, num_frms: int):
        for e in range(num_frms):
            out_dict = {}
            for key, val in data.items():
                out_dict[key] = val[e].to(self.device)

            yield out_dict


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
