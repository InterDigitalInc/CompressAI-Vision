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
        file_names = []

        self.logger.info("Processing NN-Part1...")
        for e, d in enumerate(tqdm(dataloader)):
            # TODO [hyomin - Make DefaultDatasetLoader compatible with Detectron2DataLoader]
            # Please reference to Detectron2 Dataset Mapper. Will face an issue when supporting Non-Detectron2-based network such as YOLO.

            output_file_prefix = f'img_id_{d[0]["image_id"]}'
            gt_inputs.append(
                [
                    {"image_id": d[0]["image_id"]},
                ]
            )

            file_names.append(d[0]["file_name"])

            res = self._from_input_to_features(vision_model, d, output_file_prefix)

            assert "data" in res
            num_items = self._data_buffering(res["data"])

            if e == 0:
                org_img_size = {"height": d[0]["height"], "width": d[0]["width"]}
                featureT["org_input_size"] = org_img_size
                featureT["input_size"] = res["input_size"]

                out_res = d[0].copy()
                del (
                    out_res["image"],
                    out_res["width"],
                    out_res["height"],
                    out_res["image_id"],
                )
                out_res["org_input_size"] = (d[0]["height"], d[0]["width"])
                out_res["input_size"] = featureT["input_size"][0]

        assert num_items == len(dataloader)

        if self.configs["nn_task_part1"].generate_features_only is True:
            print(
                f"features generated in {self.configs['nn_task_part1']['feature_dir']}\n exiting"
            )
            raise SystemExit(0)
        # concatenate a list of tensors at each keyword item
        featureT["data"] = self._concat_data()

        res = self._compress_features(codec, featureT, output_file_prefix)

        if self.configs["codec"]["encode_only"] is True:
            print(f"bitstreams generated, exiting")
            raise SystemExit(0)
        dec_featureT = self._decompress_features(
            codec, res["bitstream"], output_file_prefix
        )
        dec_featureT["file_name"] = d[0]["file_name"]

        # separate a tensor of each keyword item into a list of tensors
        dec_feature_tesnor = self._split_data(dec_featureT["data"])

        self.logger.info("Processing NN-Part2...")
        output_list = []
        for e, data in self._iterate_items(dec_feature_tesnor, num_items):
            dec_featureT["data"] = data
            dec_featureT["file_name"] = file_names[e]
            dec_featureT["qp"] = (
                "uncmp" if codec.qp_value is None else codec.qp_value
            )  # Assuming one qp will be used
            pred = self._from_features_to_output(vision_model, dec_featureT)
            evaluator.digest(gt_inputs[e], pred)

            out_res = dec_featureT.copy()
            del (out_res["data"], out_res["org_input_size"])
            out_res["bytes"] = res["bytes"][0]
            out_res["coded_order"] = e
            out_res["input_size"] = dec_featureT["input_size"][0]
            out_res[
                "org_input_size"
            ] = f'{dec_featureT["org_input_size"]["height"]}x{dec_featureT["org_input_size"]["width"]}'

            output_list.append(out_res)

        eval_performance = self._evaluation(evaluator)

        return codec.eval_encode_type, output_list, eval_performance

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
        for e in tqdm(range(num_frms)):
            out_dict = {}
            for key, val in data.items():
                out_dict[key] = val[e].to(self.device)

            yield e, out_dict
