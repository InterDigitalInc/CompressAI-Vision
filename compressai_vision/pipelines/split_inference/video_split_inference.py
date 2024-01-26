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


from typing import Dict, List, Tuple

import torch
from torch import Tensor
from torch.utils.data import DataLoader
from tqdm import tqdm

from compressai_vision.evaluators import BaseEvaluator
from compressai_vision.model_wrappers import BaseWrapper
from compressai_vision.registry import register_pipeline
from compressai_vision.utils import dataio, to_cpu

from ..base import BasePipeline

""" A schematic for the split-inference pipline

.. code-block:: none

             Fold                                                        Fold
           ┌ ─── ┐                                                     ┌ ─── ┐
           |     |                                                     |     |
           |     │                                                     |     │
     ┌─────┴─────▼─────┐                                         ┌─────┴─────▼─────┐
     │                 │     ┌───────────┐     ┌───────────┐     │                 │
     │     NN Task     │     │           │     │           │     │      NN Task    │
────►│                 ├────►│  Encoder  ├────►│  Decoder  ├────►│                 ├────►
     │      Part 1     │     │           │     │           │     │      Part 2     │
     │                 │     └───────────┘     └───────────┘     │                 │
     └─────────────────┘                                         └─────────────────┘

"""


@register_pipeline("video-split-inference")
class VideoSplitInference(BasePipeline):
    def __init__(
        self,
        configs: Dict,
        device: str,
    ):
        super().__init__(configs, device)

        self._input_ftensor_buffer = []

    def build_input_lists(self, dataloader: DataLoader) -> Tuple[List]:
        gt_inputs = []
        file_names = []
        for _, d in enumerate(dataloader):
            gt_inputs.append(
                [
                    {"image_id": d[0]["image_id"]},
                ]
            )
            file_names.append(d[0]["file_name"])
        return gt_inputs, file_names

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
        self._update_codec_configs_at_pipeline_level(len(dataloader))

        features = {}
        gt_inputs, file_names = self.build_input_lists(dataloader)

        timing = {"nn_part_1": 0, "encode": 0, "decode": 0, "nn_part_2": 0}

        if not self.configs["codec"]["decode_only"]:
            ## NN-part-1
            for e, d in enumerate(tqdm(dataloader)):
                output_file_prefix = f'img_id_{d[0]["image_id"]}'

                if e < self._codec_skip_n_frames:
                    continue
                if e >= self._codec_end_frame_idx:
                    break

                start = self.time_measure()
                res = self._from_input_to_features(vision_model, d, output_file_prefix)
                end = self.time_measure()
                timing["nn_part_1"] = timing["nn_part_1"] + (end - start)

                assert "data" in res

                num_items = self._input_data_collecting(res["data"])

                if (e - self._codec_skip_n_frames) == 0:
                    org_img_size = {"height": d[0]["height"], "width": d[0]["width"]}
                    features["org_input_size"] = org_img_size
                    features["input_size"] = res["input_size"]

                    out_res = d[0].copy()
                    del (
                        out_res["image"],
                        out_res["width"],
                        out_res["height"],
                        out_res["image_id"],
                    )
                    out_res["org_input_size"] = (d[0]["height"], d[0]["width"])
                    out_res["input_size"] = features["input_size"][0]

            assert num_items == self._codec_n_frames_to_be_encoded

            if self.configs["nn_task_part1"].generate_features_only is True:
                print(
                    f"features generated in {self.configs['nn_task_part1']['feature_dir']}\n exiting"
                )
                raise SystemExit(0)

            # concatenate a list of tensors at each keyword item
            features["data"] = self._reform_list_to_dict(self._input_ftensor_buffer)

            # Feature Compression
            start = self.time_measure()
            res = self._compress_features(
                codec, features, self.codec_output_dir, self.bitstream_name, ""
            )
            end = self.time_measure()
            timing["encode"] = timing["encode"] + (end - start)

            if self.configs["codec"]["encode_only"] is True:
                print(f"bitstreams generated, exiting")
                raise SystemExit(0)
        else:  # decode only
            res = {}
            bin_files = [
                file_path
                for file_path in self.codec_output_dir.glob(f"{self.bitstream_name}*")
                if file_path.suffix in [".bin", ".mp4"]
            ]
            assert (
                len(bin_files) > 0
            ), f"no bitstream file matching {self.bitstream_name}*"
            assert (
                len(bin_files) == 1
            ), f"Error, multiple bitstream files matching {self.bitstream_name}*"
            res["bitstream"] = bin_files[0]
            bitstream_bytes = res["bitstream"].stat().st_size

        # Feature Deompression
        start = self.time_measure()
        dec_features = self._decompress_features(
            codec, res["bitstream"], self.codec_output_dir, ""
        )
        end = self.time_measure()
        timing["decode"] = timing["decode"] + (end - start)

        # dec_features should contain "org_input_size" and "input_size"
        # When using anchor codecs, that's not the case, we read input images to derive them
        if not "org_input_size" in dec_features or not "input_size" in dec_features:
            self.logger.warning(
                "Hacky: 'org_input_size' and 'input_size' retrived from input dataset."
            )
            first_frame = next(iter(dataloader))
            org_img_size = {
                "height": first_frame[0]["height"],
                "width": first_frame[0]["width"],
            }
            dec_features["org_input_size"] = org_img_size
            dec_features["input_size"] = self._get_model_input_size(
                vision_model, first_frame
            )

        # separate a tensor of each keyword item into a list of tensors
        dec_ftensors_list = self._reform_dict_to_list(dec_features["data"])
        assert len(dataloader) == len(
            dec_ftensors_list
        ), "The number of decoded frames are not equal to the number of frames supposed to be decoded"

        self.logger.info("Processing NN-Part2...")
        output_list = []
        for e, data in tqdm(self._iterate_items(dec_ftensors_list, self.device)):
            dec_features["data"] = data
            dec_features["file_name"] = file_names[e]
            dec_features["qp"] = (
                "uncmp" if codec.qp_value is None else codec.qp_value
            )  # Assuming one qp will be used

            start = self.time_measure()
            pred = self._from_features_to_output(vision_model, dec_features)
            end = self.time_measure()

            timing["nn_part_2"] = timing["nn_part_2"] + (end - start)

            evaluator.digest(gt_inputs[e], pred)

            out_res = dec_features.copy()
            del (out_res["data"], out_res["org_input_size"])
            if self.configs["codec"]["decode_only"]:
                out_res["bytes"] = bitstream_bytes / len(dataloader)
            else:
                out_res["bytes"] = res["bytes"][e]
            out_res["coded_order"] = e
            out_res["input_size"] = dec_features["input_size"][0]
            out_res["org_input_size"] = (
                f'{dec_features["org_input_size"]["height"]}x{dec_features["org_input_size"]["width"]}'
            )

            output_list.append(out_res)

        # performance evaluation on end-task
        eval_performance = self._evaluation(evaluator)

        return timing, codec.eval_encode_type, output_list, eval_performance

    def _input_data_collecting(self, data: Dict):
        """
        Pull up input data and move to generic memory in case it was on GPU memory
        """

        d = {}
        for key, tensor in data.items():
            d[key] = to_cpu(tensor)

        del data
        self._input_ftensor_buffer.append(d)

        return len(self._input_ftensor_buffer)

    @staticmethod
    def _reform_list_to_dict(data: List):
        output = None

        for ftensors in data:
            assert isinstance(ftensors, dict)

            if output is None:
                output = dict(
                    zip(ftensors.keys(), [[] for _ in range(len(ftensors.keys()))])
                )

            for key, ftensor in ftensors.items():
                output[key].append(ftensor)

        for key, ftensors in output.items():
            output[key] = torch.concat(ftensors, dim=0)

        return output

    @staticmethod
    def _reform_dict_to_list(data: Dict):
        output = []
        tmp = {}
        total_len = -1
        for key, tensor in data.items():
            if isinstance(tensor, Tensor):
                tmp[key] = list(tensor.chunk(len(tensor)))
            elif isinstance(tensor, List):
                tmp[key] = tensor
            else:
                raise NotImplementedError

            total_len = len(tmp[key])

        assert all(len(item) == total_len for item in tmp.values())

        keys = list(data.keys())
        for ftensors in zip(*(tmp.values())):
            d = dict(zip(keys, ftensors))
            output.append(d)

        return output

    @staticmethod
    def _iterate_items(data: List, device):
        for e, ftensors in enumerate(tqdm(data)):
            out_dict = {}
            for key, val in ftensors.items():
                out_dict[key] = val.to(device)

            yield e, out_dict
