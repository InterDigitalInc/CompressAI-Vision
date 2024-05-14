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


import logging
import os
import shutil
from enum import Enum
from glob import glob
from pathlib import Path
from typing import Callable, Dict, List
from uuid import uuid4 as uuid

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from compressai_vision.evaluators import BaseEvaluator
from compressai_vision.model_wrappers import BaseWrapper
from compressai_vision.registry import register_pipeline
from compressai_vision.utils import dataio, time_measure

from ..base import BasePipeline

""" An example schematic for the single input multiple tasks pipline

.. code-block:: none

                                     ┌───────────────────────┐
                                     │                       │
                                     │      NN Task 3        │
                          ┌─────────▶│       Part 2          │
                          │          │ (i.e., Pixel Decoder) │
                          │          └───────────────────────┘
 ┌────────────────┐       │          ┌───────────────────────┐
 │                │       │          │                       │
 │    Encoder     ├───────┘          │      NN Task 2        │
 │                ├─────────────────▶│       Part 2          │
 │                ├────────────┐     │ (i.e., Segmentation)  │
 └────────────────┘            │     └───────────────────────┘
                               │     ┌───────────────────────┐
                               │     │                       │
                               │     │      NN Task 1        │
                               └────▶│       Part 2          │
                                     │ (i.e., Obj. Detection)│
                                     └───────────────────────┘

"""


@register_pipeline("multi-task-inference")
class MultiTaskInference(BasePipeline):
    def __init__(
        self,
        configs: Dict,
        device: str,
    ):
        super().__init__(configs, device)

        self.target_task_layer_id = configs["codec"].target_task_layer
        self.num_tasks = configs["codec"].num_tasks
        self.init_time_measure()

    def init_time_measure(self):
        self.elapsed_time = {"encode": 0, "decode": 0, "nn_part_2": 0}

    def __call__(
        self,
        codec,
        vision_models: List,
        dataloader: DataLoader,
        evaluators: List,
    ) -> Dict:
        """Push image(s) through the encoder+decoder, returns number of bits for each image and encoded+decoded images

        Returns (nbitslist, x_hat), where nbitslist is a list of number of bits and x_hat is the image that has gone throught the encoder/decoder process
        """

        assert (
            codec.num_tasks == len(evaluators) == len(vision_models)
        ), f"# of multiple tasks are not matched"

        self._update_codec_configs_at_pipeline_level(len(dataloader))
        output_list = []

        tlid = self.target_task_layer_id
        for e, d in enumerate(tqdm(dataloader)):
            file_prefix = f'img_id_{d[0]["image_id"]}'

            if not self.configs["codec"]["decode_only"]:
                if e < self._codec_skip_n_frames:
                    continue
                if e >= self._codec_end_frame_idx:
                    break

                start = time_measure()
                res = self._compress(
                    codec,
                    d[0],
                    self.codec_output_dir,
                    self.bitstream_name,
                    file_prefix,
                )
                self.update_time_elapsed("encode", (time_measure() - start))
            else:
                res = {}
                bitstream_name = f"{self.bitstream_name}-{file_prefix}"
                bitstream_path = os.path.join(self.codec_output_dir, bitstream_name)
                bitstream_lists = sorted(glob(f"{bitstream_path}_*.bin"))

                assert len(bitstream_lists) <= self.num_tasks and len(
                    bitstream_lists
                ) >= (tlid + 1)

                bitstream_lists = bitstream_lists[: (tlid + 1)]

                print(f"reading bitstream... {bitstream_lists}")
                res["bitstream"] = bitstream_lists

            if self.configs["codec"]["encode_only"] is True:
                continue

            start = time_measure()
            dec_features = self._decompress(
                codec, res["bitstream"], self.codec_output_dir, file_prefix
            )
            self.update_time_elapsed("decode", (time_measure() - start))

            assert "input_size" in dec_features
            assert "org_input_size" in dec_features

            dec_features["file_name"] = d[0]["file_name"]

            start = time_measure()
            preds = self._from_features_to_output(
                vision_models, dec_features, file_prefix
            )
            self.update_time_elapsed("nn_part_2", (time_measure() - start))

            # assert len(evaluators) == len(preds)
            evaluators[tlid].digest(d, preds[tlid])

            out_res = d[0].copy()
            del (
                out_res["image"],
                out_res["width"],
                out_res["height"],
                out_res["image_id"],
            )
            out_res["qp"] = (
                "uncmp" if codec.qp_value is None else codec.qp_value
            )  # Assuming one qp will be used
            if self.configs["codec"]["decode_only"]:
                acbytes = 0
                for fpath in res["bitstream"]:
                    acbytes += os.stat(fpath).st_size
                out_res["bytes"] = acbytes
            else:
                out_res["bytes"] = res["bytes"][0]
            out_res["coded_order"] = e
            out_res["org_input_size"] = f'{d[0]["height"]}x{d[0]["width"]}'
            out_res["input_size"] = dec_features["input_size"][0]
            output_list.append(out_res)

        if self.configs["codec"]["encode_only"] is True:
            print(f"bitstreams generated, exiting")
            raise SystemExit(0)

        eval_performance = self._evaluation(evaluators[tlid])

        return (
            self.time_elapsed_by_module,
            codec.eval_encode_type,
            output_list,
            eval_performance,
        )

    def _compress(self, codec, x, codec_output_dir, bitstream_name, filename: str):
        return codec.encode(
            x,
            codec_output_dir,
            bitstream_name,
            filename,
        )

    def _decompress(self, codec, bitstream, codec_output_dir: str, filename: str):
        return codec.decode(
            bitstream,
            codec_output_dir,
            filename,
        )

    def _from_features_to_output(
        self, vision_models: list, x: Dict, seq_name: str = None
    ):
        """performs the inference of the 2nd part of the NN model"""

        assert "data" in x

        output_results_dir = self.configs["nn_task_part2"].output_results_dir

        results = []
        for tid, val in enumerate(x["data"].values()):
            results_file = f"{output_results_dir}/{seq_name}_tl{tid}{self._output_ext}"

            if vision_models[tid] is not None:
                in_x = x.copy()
                del in_x["data"]

                # suppose that the order of keys and values is matched
                in_x["data"] = {
                    k: v.to(device=self.device)
                    for k, v in zip(vision_models[tid].split_layer_list, [val])
                }

                if tid >= self.target_task_layer_id:  # task validity:
                    out = vision_models[tid].features_to_output(in_x)
                    results.append(out)

                    if self.configs["nn_task_part2"].dump_results:
                        self._create_folder(output_results_dir)
                        torch.save(out, results_file)
                else:
                    results.append(None)
            else:  # output is a decoded output
                assert val.dim() == 4 and val.shape[0] == 1
                out = val.squeeze(0)
                results.append(out)

        return results
