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


from typing import Dict, List, Tuple, TypeVar

import torch
from torch import Tensor
from torch.utils.data import DataLoader
from tqdm import tqdm

from compressai_vision.evaluators import BaseEvaluator
from compressai_vision.model_wrappers import BaseWrapper
from compressai_vision.registry import register_pipeline
from compressai_vision.utils import dl_to_ld, ld_to_dl, time_measure, to_cpu
from compressai_vision.utils.measure_complexity import (
    calc_complexity_nn_part1_dn53,
    calc_complexity_nn_part1_plyr,
    calc_complexity_nn_part2_dn53,
    calc_complexity_nn_part2_plyr,
)

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

K = TypeVar("K")
V = TypeVar("V")


@register_pipeline("video-split-inference")
class VideoSplitInference(BasePipeline):
    def __init__(
        self,
        configs: Dict,
        device: Dict,
    ):
        super().__init__(configs, device)
        self._input_ftensor_buffer = []
        self.datatype = configs["datatype"]

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
        """
        Processes input data with the split inference video pipeline: compresses features, decompresses features, and evaluates performance.

        Args:
            vision_model (BaseWrapper): The vision model wrapper.
            codec: The codec used for compression.
            dataloader (DataLoader): The data loader for input data.
            evaluator (BaseEvaluator): The evaluator used for performance evaluation.

        Returns:
            Dict: A dictionary containing timing information, codec evaluation type, a list of output results, and performance evaluation metrics.
        """

        self._update_codec_configs_at_pipeline_level(len(dataloader))

        features = {}
        gt_inputs, file_names = self.build_input_lists(dataloader)

        self.init_time_measure()
        self.init_complexity_measure()

        if not self.configs["codec"]["decode_only"]:
            ## NN-part-1
            for e, d in enumerate(tqdm(dataloader)):
                output_file_prefix = f'img_id_{d[0]["image_id"]}'

                if e < self._codec_skip_n_frames:
                    continue
                if e >= self._codec_end_frame_idx:
                    break

                if self.is_mac_calculation and e == self._codec_skip_n_frames:
                    if hasattr(vision_model, "darknet"):  # for jde
                        kmacs, pixels = calc_complexity_nn_part1_dn53(vision_model, d)
                    else:  # for detectron2
                        kmacs, pixels = calc_complexity_nn_part1_plyr(vision_model, d)
                    self.add_kmac_and_pixels_info("nn_part_1", kmacs, pixels)

                start = time_measure()
                res = self._from_input_to_features(
                    vision_model, d, output_file_prefix, evaluator.datacatalog_name
                )
                self.update_time_elapsed("nn_part_1", (time_measure() - start))

                self._input_ftensor_buffer.append(
                    {k: to_cpu(tensor) for k, tensor in res["data"].items()}
                )

                del res["data"]

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

                del d[0]["image"]

            assert len(self._input_ftensor_buffer) == self._codec_n_frames_to_be_encoded

            if self.configs["nn_task_part1"].generate_features_only is True:
                print(
                    f"features generated in {self.configs['nn_task_part1']['feature_dir']}\n exiting"
                )
                raise SystemExit(0)

            # concatenate a list of tensors at each keyword item
            features["data"] = self._feature_tensor_list_to_dict(
                self._input_ftensor_buffer
            )
            self._input_ftensor_buffer = []

            # datatype conversion
            features["data"] = {
                k: v.type(getattr(torch, self.datatype))
                for k, v in features["data"].items()
            }

            # Feature Compression
            start = time_measure()
            res, enc_time_by_module, enc_complexity = self._compress(
                codec, features, self.codec_output_dir, self.bitstream_name, ""
            )
            self.update_time_elapsed("encode", (time_measure() - start))
            self.add_time_details("encode", enc_time_by_module)
            if self.is_mac_calculation:
                self.add_kmac_and_pixels_info(
                    "feature_reduction", enc_complexity[0], enc_complexity[1]
                )

            # for bypass mode, 'data' should be deleted.
            if "data" in res["bitstream"] is False:
                del features["data"]

            if self.configs["codec"]["encode_only"] is True:
                print("bitstreams generated, exiting")
                raise SystemExit(0)

        else:  # decode only
            res = {}
            bin_files = [
                file_path
                for file_path in self.codec_output_dir.glob(f"{self.bitstream_name}*")
                if (
                    (file_path.suffix in [".bin", ".mp4"])
                    and not "_tmp" in file_path.name
                )
            ]
            assert (
                len(bin_files) > 0
            ), f"Error: decode_only mode, no bitstream file matching {self.bitstream_name}*"
            assert (
                len(bin_files) == 1
            ), f"Error, decode_only mode, multiple bitstream files matching {self.bitstream_name}*"
            res["bitstream"] = bin_files[0]
            bitstream_bytes = res["bitstream"].stat().st_size

        # Feature Deompression
        start = time_measure()
        dec_features, dec_time_by_module, dec_complexity = self._decompress(
            codec, res["bitstream"], self.codec_output_dir, ""
        )
        self.update_time_elapsed("decode", (time_measure() - start))
        self.add_time_details("decode", dec_time_by_module)
        if self.is_mac_calculation:
            self.add_kmac_and_pixels_info(
                "feature_restoration", dec_complexity[0], dec_complexity[1]
            )

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
        dec_ftensors_list = self._feature_tensor_dict_to_list(dec_features["data"])
        assert all(
            [self.datatype in str(d.dtype) for d in dec_ftensors_list[0].values()]
        ), "Output features not of expected datatype"

        dec_ftensors_list = [
            {k: v.type(torch.float32) for k, v in d.items()} for d in dec_ftensors_list
        ]

        assert len(dec_ftensors_list) == len(dataloader), (
            f"The number of decoded frames ({len(dec_ftensors_list)}) is not equal "
            f"to the number of frames supposed to be decoded ({len(dataloader)})"
        )

        self.logger.info("Processing NN-Part2...")
        output_list = []

        for e, ftensors in enumerate(tqdm(dec_ftensors_list)):
            data = {k: v.to(self.device_nn_part2) for k, v in ftensors.items()}
            dec_features["data"] = data
            dec_features["file_name"] = file_names[e]
            dec_features["qp"] = (
                "uncmp" if codec.qp_value is None else codec.qp_value
            )  # Assuming one qp will be used

            if self.is_mac_calculation and e == 0:
                if hasattr(vision_model, "darknet"):  # for jde
                    kmacs, pixels = calc_complexity_nn_part2_dn53(
                        vision_model, dec_features
                    )
                else:  # for detectron2
                    kmacs, pixels = calc_complexity_nn_part2_plyr(
                        vision_model, data, dec_features
                    )
                self.add_kmac_and_pixels_info("nn_part_2", kmacs, pixels)

            start = time_measure()
            pred = self._from_features_to_output(vision_model, dec_features)
            self.update_time_elapsed("nn_part_2", (time_measure() - start))

            if evaluator:
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

        # Calculate mac considering number of coded feature frames
        if self.is_mac_calculation:
            frames = (
                len(dataloader) // 2 + 1
                if codec.enc_tools["feature_reduction"]["temporal_resampling_enabled"]
                is True
                else len(dataloader)
            )
            self.calc_kmac_per_pixels_video_task(frames, len(dataloader))

        # performance evaluation on end-task
        eval_performance = self._evaluation(evaluator)

        return (
            self.time_elapsed_by_module,
            codec.eval_encode_type,
            output_list,
            eval_performance,
            self.complexity_calc_by_module,
        )

    @staticmethod
    def _feature_tensor_list_to_dict(
        data: List[Dict[str, Tensor]]
    ) -> Dict[str, Tensor]:
        """
        Converts a list of feature tensors into a dictionary format.
        """
        return {k: torch.cat(ftensors) for k, ftensors in ld_to_dl(data).items()}

    @staticmethod
    def _feature_tensor_dict_to_list(data: Dict):
        """
        Converts a dict of feature tensors into a list of tensors.
        """

        def coerce_tensors(vs) -> List[Tensor]:
            assert isinstance(vs, Tensor) or (
                isinstance(vs, list) and all(isinstance(v, Tensor) for v in vs)
            )
            return list(vs)

        return dl_to_ld({k: coerce_tensors(tensors) for k, tensors in data.items()})
