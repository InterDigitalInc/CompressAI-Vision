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

import os
import sys

from typing import Dict

from torch.utils.data import DataLoader
from tqdm import tqdm

from compressai_vision.evaluators import BaseEvaluator
from compressai_vision.model_wrappers import BaseWrapper
from compressai_vision.registry import register_pipeline
from compressai_vision.utils import metric_tracking, time_measure

from ..base import BasePipeline

""" A schematic for the remote-inference pipline

.. code-block:: none

                                           ┌─────────────────┐
     ┌───────────┐       ┌───────────┐     │                 │
     │           │       │           │     │      NN Task    │
────►│  Encoder  ├──────►│  Decoder  ├────►│                 ├────►
     │           │       │           │     │                 │
     └───────────┘       └───────────┘     │                 │
                                           └─────────────────┘
                         <---------------- Remote Server ------------->
──►──────►──────►────────►──────►──────►──────►──────►──────►──────►──────►
"""


@register_pipeline("image-remote-inference")
class ImageRemoteInference(BasePipeline):
    def __init__(
        self,
        configs: Dict,
        device: str,
    ):
        super().__init__(configs, device)

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
        org_map_func = dataloader.dataset.get_org_mapper_func()
        output_list = []
        timing = {
            "encode": metric_tracking(),
            "decode": metric_tracking(),
            "nn_task": metric_tracking(),
        }

        for e, d in enumerate(tqdm(dataloader)):
            org_img_size = {"height": d[0]["height"], "width": d[0]["width"]}
            file_prefix = f'img_id_{d[0]["image_id"]}'

            if not self.configs["codec"]["decode_only"]:
                if e < self._codec_skip_n_frames:
                    continue
                if e >= self._codec_end_frame_idx:
                    break

                start = time_measure()
                frame = {
                    "file_names": [d[0]["file_name"]],
                    "org_input_size": org_img_size,
                }

                res, enc_time_details, _ = self._compress(
                    codec,
                    frame,
                    self.codec_output_dir,
                    self.bitstream_name,
                    file_prefix,
                    remote_inference=True,
                )
                end = time_measure()
                timing["encode"].append((end - start))
            else:
                res = {}
                bin_files = [
                    file_path
                    for file_path in self.codec_output_dir.glob(
                        f"{self.bitstream_name}-{file_prefix}*"
                    )
                    if file_path.suffix in [".bin", ".mp4"]
                ]
                assert (
                    len(bin_files) > 0
                ), f"no bitstream file matching {self.bitstream_name}-{file_prefix}*"
                assert (
                    len(bin_files) == 1
                ), f"Error, multiple bitstream files matching {self.bitstream_name}*"

                res["bitstream"] = bin_files[0]
                print(f"reading bitstream... {res['bitstream']}", file=sys.stdout)

            if self.configs["codec"]["encode_only"] is True:
                continue

            start = time_measure()
            dec_seq, dec_time_by_module, mac_computation = self._decompress(
                codec,
                res["bitstream"],
                self.codec_output_dir,
                file_prefix,
                org_img_size,
                remote_inference=True,
                vcm_mode=self.configs["codec"]["vcm_mode"],
            )
            end = time_measure()
            timing["decode"].append((end - start))

            start = time_measure()
            dec_d = {"file_name": dec_seq["file_names"][0]}
            # dec_d = {"file_name": dec_seq[0]["file_names"][0]}

            pred = vision_model.forward(org_map_func(dec_d))
            end = time_measure()
            timing["nn_task"].append((end - start))

            if getattr(self, "vis_dir", None) and hasattr(
                evaluator, "save_visualization"
            ):
                evaluator.save_visualization(d, pred, self.vis_dir, self.vis_threshold)

            evaluator.digest(d, pred)

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

            if not isinstance(res["bitstream"], dict):
                out_res["bytes"] = os.stat(res["bitstream"]).st_size
            else:
                out_res["bytes"] = res["bytes"][0]

            out_res["coded_order"] = e
            out_res["org_input_size"] = f'{d[0]["height"]}x{d[0]["width"]}'
            output_list.append(out_res)

        if self.configs["codec"]["encode_only"] is True:
            print("bitstreams generated, exiting")
            raise SystemExit(0)

        eval_performance = self._evaluation(evaluator)

        for key, val in timing.items():
            timing[key] = val.sum

        return timing, codec.eval_encode_type, output_list, eval_performance


# Please leave this function for reference
def debugging(file_prefix, d, old_file_name):
    import math

    from compressai_vision.utils.external_exec import run_cmdline

    dst = "local/folder"
    # jpg --> yuv
    frame_width = math.ceil(d[0]["width"] / 2) * 2
    frame_height = math.ceil(d[0]["height"] / 2) * 2
    file_prefix = f"{file_prefix}_{frame_width}x{frame_height}_1fps_10bit_pyuv420"
    yuv_in_path = f"{file_prefix}_input.yuv"

    convert_cmd = [
        "ffmpeg",
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        old_file_name,
        "-vf",
        "pad=ceil(iw/2)*2:ceil(ih/2)*2",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "yuv420p10le",
        "-dst_range",
        "1",  #  (fracape) convert to full range for now
        f"{dst}/{yuv_in_path}",
    ]

    run_cmdline(convert_cmd)

    # yuv --> png
    output_file_prefix = f"{file_prefix}_tmp.png"
    convert_cmd = [
        "ffmpeg",
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "yuv420p10le",
        "-s",
        f"{frame_width}x{frame_height}",
        "-src_range",
        "1",  # (fracape) assume dec yuv is full range for now
        "-i",
        f"{dst}/{yuv_in_path}",
        "-pix_fmt",
        "rgb24",
        f"{dst}/{output_file_prefix}",
    ]

    run_cmdline(convert_cmd)

    final_png = f"{file_prefix}.png"
    convert_cmd = [
        "ffmpeg",
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        f"{dst}/{output_file_prefix}",
        "-vf",
        f"crop={d[0]['width']}:{d[0]['height']}",
        f"{dst}/{final_png}",  # no name change
    ]
    run_cmdline(convert_cmd)

    return {"file_name": f"{dst}/{final_png}"}
