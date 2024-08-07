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
from typing import Dict

from torch.utils.data import DataLoader
from tqdm import tqdm

from compressai_vision.evaluators import BaseEvaluator
from compressai_vision.model_wrappers import BaseWrapper
from compressai_vision.registry import register_pipeline
from compressai_vision.utils import dict_sum, time_measure
from compressai_vision.utils.measure_complexity import calc_complexity_nn_part1_plyr, calc_complexity_nn_part2_plyr, calc_complexity_nn_part1_dn53, calc_complexity_nn_part2_dn53

from ..base import BasePipeline

""" A schematic for the split-inference pipline

.. code-block:: none

     ┌─────────────────┐                                         ┌─────────────────┐
     │                 │     ┌───────────┐     ┌───────────┐     │                 │
     │     NN Task     │     │           │     │           │     │      NN Task    │
────►│                 ├────►│  Encoder  ├────►│  Decoder  ├────►│                 ├────►
     │      Part 1     │     │           │     │           │     │      Part 2     │
     │                 │     └───────────┘     └───────────┘     │                 │
     └─────────────────┘                                         └─────────────────┘

    ──────►──────►──────►──────►──────►──────►──────►──────►──────►──────►
"""


@register_pipeline("image-split-inference")
class ImageSplitInference(BasePipeline):
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
        """
        Processes input data with the split inference image pipeline: compresses features, decompresses features, and evaluates performance.

        Args:
            vision_model (BaseWrapper): The vision model wrapper.
            codec: The codec used for compression.
            dataloader (DataLoader): The data loader for input data.
            evaluator (BaseEvaluator): The evaluator used for performance evaluation.

        Returns:
            Dict: A dictionary containing timing information, codec evaluation type, a list of output results, and performance evaluation metrics.
        """
        self._update_codec_configs_at_pipeline_level(len(dataloader))
        output_list = []

        self.init_time_measure()
        self.init_complexity_measure()
        accum_enc_by_module = None
        accum_dec_by_module = None

        for e, d in enumerate(tqdm(dataloader)):
            org_img_size = {"height": d[0]["height"], "width": d[0]["width"]}
            file_prefix = f'img_id_{d[0]["image_id"]}'

            if not self.configs["codec"]["decode_only"]:
                if e < self._codec_skip_n_frames:
                    continue
                if e >= self._codec_end_frame_idx:
                    break
                
                if self.is_mac_calculation:
                    macs = calc_complexity_nn_part1_plyr(vision_model, d)                    
                    self.calc_total_kmac_image_task("nn_part_1", macs)
                
                start = time_measure()
                featureT = self._from_input_to_features(vision_model, d, file_prefix)
                self.update_time_elapsed("nn_part_1", (time_measure() - start))

                featureT["org_input_size"] = org_img_size

                start = time_measure()
                res, enc_time_by_module, enc_complexity = self._compress(
                    codec,
                    featureT,
                    self.codec_output_dir,
                    self.bitstream_name,
                    file_prefix,
                )
                self.update_time_elapsed("encode", (time_measure() - start))
                if self.is_mac_calculation:
                    self.calc_total_kmac_image_task("feature_reduction", enc_complexity)
                
                if accum_enc_by_module is None:
                    accum_enc_by_module = enc_time_by_module
                else:
                    accum_enc_by_module = dict_sum(
                        accum_enc_by_module, enc_time_by_module
                    )
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
                print(f"reading bitstream... {res['bitstream']}")

            if self.configs["codec"]["encode_only"] is True:
                continue

            start = time_measure()
            dec_features, dec_time_by_module, dec_complexity = self._decompress(
                codec, res["bitstream"], self.codec_output_dir, file_prefix
            )
            self.update_time_elapsed("decode", (time_measure() - start))
            if self.is_mac_calculation:
                self.calc_total_kmac_image_task("feature_restoration", dec_complexity)
            
            if accum_dec_by_module is None:
                accum_dec_by_module = dec_time_by_module
            else:
                accum_dec_by_module = dict_sum(accum_dec_by_module, dec_time_by_module)

            # dec_features should contain "org_input_size" and "input_size"
            # When using anchor codecs, that's not the case, we read input images to derive them
            if not "org_input_size" in dec_features or not "input_size" in dec_features:
                self.logger.warning(
                    "Hacky: 'org_input_size' and 'input_size' retrived from input dataset."
                )
                dec_features["org_input_size"] = org_img_size
                dec_features["input_size"] = self._get_model_input_size(vision_model, d)

            dec_features["file_name"] = d[0]["file_name"]
            if self.is_mac_calculation:
                macs = calc_complexity_nn_part2_plyr(vision_model, dec_features["data"], dec_features)
                self.calc_total_kmac_image_task("nn_part_2", macs)

            start = time_measure()
            pred = self._from_features_to_output(
                vision_model, dec_features, file_prefix
            )
            self.update_time_elapsed("nn_part_2", (time_measure() - start))

            if evaluator:
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
            if self.configs["codec"]["decode_only"]:
                out_res["bytes"] = os.stat(res["bitstream"]).st_size
            else:
                out_res["bytes"] = res["bytes"][0]
            out_res["coded_order"] = e
            out_res["org_input_size"] = f'{d[0]["height"]}x{d[0]["width"]}'
            out_res["input_size"] = dec_features["input_size"][0]
            output_list.append(out_res)

        # if dec_only is True, accum_enc_by_module is None
        self.add_time_details("encode", accum_enc_by_module)
        # if enc_only is True, accum_dec_by_module is None
        self.add_time_details("decode", accum_dec_by_module)

        if self.configs["codec"]["encode_only"] is True:
            print(f"bitstreams generated, exiting")
            return self.time_elapsed_by_module, codec.eval_encode_type, None, None

        eval_performance = self._evaluation(evaluator)

        return (
            self.time_elapsed_by_module,
            codec.eval_encode_type,
            output_list,
            eval_performance,
            self.complexity_calc_by_module,
        )
