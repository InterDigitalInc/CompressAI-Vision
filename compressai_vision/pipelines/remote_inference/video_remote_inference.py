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

from pathlib import Path
from typing import Dict, List, Tuple

import torch

from PIL import Image
from torch import Tensor
from torch.utils.data import DataLoader
from tqdm import tqdm

from compressai_vision.evaluators import BaseEvaluator
from compressai_vision.model_wrappers import BaseWrapper
from compressai_vision.registry import register_pipeline
from compressai_vision.utils import metric_tracking, time_measure, to_cpu

from ..base import BasePipeline

""" A schematic for the remote-inference pipline

.. code-block:: none


          Fold
        ┌ ─── ┐
        |     |
        |     │                            ┌─────────────────┐
     ┌──┴─────▼──┐       ┌───────────┐     │                 │
     │           │       │           │     │      NN Task    │
────►│  Encoder  ├──────►│  Decoder  ├────►│                 ├────►
     │           │       │           │     │                 │
     └───────────┘       └───────────┘     │                 │
                                           └─────────────────┘
                         <---------------- Remote Server ------------->
──►──────►──────►────────►──────►──────►──────►──────►──────►──────►──────►

"""


@register_pipeline("video-remote-inference")
class VideoRemoteInference(BasePipeline):
    def __init__(
        self,
        configs: Dict,
        device: str,
    ):
        super().__init__(configs, device)

        self._input_ftensor_buffer = []
        self._video_yuv = configs["yuv"] if "yuv" in configs else None

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

        gt_inputs, file_names = self.build_input_lists(dataloader)

        timing = {
            "encode": metric_tracking(),
            "decode": metric_tracking(),
            "nn_task": metric_tracking(),
        }

        frames = {}
        if not self.configs["codec"]["decode_only"]:
            width, height = Image.open(file_names[0]).size
            org_input_size = {
                "height": height,
                "width": width,
            }
            frames = {
                "frame_skip": self._codec_skip_n_frames,
                "last_frame": self._codec_end_frame_idx,
                "file_names": file_names,
                "org_input_size": org_input_size,
            }

            start = time_measure()
            res, enc_time_by_module, enc_complexity = self._compress(
                codec,
                frames,
                self.codec_output_dir,
                self.bitstream_name,
                "",
                remote_inference=True,
            )
            end = time_measure()
            timing["encode"].append((end - start))

            if self.configs["codec"]["encode_only"] is True:
                print("bitstreams generated, exiting")
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
            # bitstream_bytes = res["bitstream"].stat().st_size

        # Feature Deompression
        start = time_measure()
        dec_seq, dec_time_by_module, dec_complexity = self._decompress(
            codec=codec,
            bitstream=res["bitstream"],
            codec_output_dir=self.codec_output_dir,
            filename="",  # must be empty like this
            org_img_size=None,
            remote_inference=True,
            vcm_mode=self.configs["codec"]["vcm_mode"],
        )
        end = time_measure()
        timing["decode"].append((end - start))

        self.logger.info("Processing remote NN")

        output_list = []
        org_map_func = dataloader.dataset.get_org_mapper_func()
        for e, d in enumerate(tqdm(dataloader)):
            # some assertion needed to check if d is matched with dec_seq[e]

            start = time_measure()
            dec_d = {"file_name": dec_seq["file_names"][e]}
            # dec_d = {"file_name": dec_seq[0]["file_names"][e]}
            pred = vision_model.forward(org_map_func(dec_d))
            end = time_measure()
            timing["nn_task"].append((end - start))

            if getattr(self, "vis_dir", None) and hasattr(
                evaluator, "save_visualization"
            ):
                evaluator.save_visualization(d, pred, self.vis_dir, self.vis_threshold)

            evaluator.digest(d, pred)

            out_res = d[0].copy()
            del out_res["image"]
            out_res["qp"] = (
                "uncmp" if codec.qp_value is None else codec.qp_value
            )  # Assuming one qp will be used

            if not isinstance(res["bitstream"], dict):
                out_res["bytes"] = Path(res["bitstream"]).stat().st_size / len(
                    dataloader
                )
            else:
                assert len(res["bytes"]) == len(dataloader)
                out_res["bytes"] = res["bytes"][e]

            out_res["coded_order"] = e
            out_res["org_input_size"] = f'{d[0]["height"]}x{d[0]["width"]}'

            output_list.append(out_res)

        # performance evaluation on end-task
        eval_performance = self._evaluation(evaluator)

        for key, val in timing.items():
            timing[key] = val.sum

        return timing, codec.eval_encode_type, output_list, eval_performance
