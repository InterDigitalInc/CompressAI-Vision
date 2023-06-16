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

import json

from detectron2.evaluation import COCOEvaluator

from compressai_vision.datasets import deccode_compressed_rle
from compressai_vision.registry import register_evaluator

from .base_evaluator import BaseEvaluator


@register_evaluator("COCO-EVAL")
class COCOEVal(BaseEvaluator):
    def __init__(self, datacatalog_name, dataset_name, output_dir="./vision_output/"):
        super().__init__(datacatalog_name, dataset_name, output_dir)

        self._evaluator = COCOEvaluator(
            dataset_name, False, output_dir=output_dir, use_fast_impl=False
        )

        if datacatalog_name == "MPEGOIV6":
            deccode_compressed_rle(self._evaluator._coco_api.anns)

        self._evaluator.reset()

    def digest(self, gt, pred):
        return self._evaluator.process(gt, pred)

    def results(self, save_path: str = None):
        out = self._evaluator.evaluate()

        file_path = f"{save_path}/results.json"
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(out, f, ensure_ascii=False, indent=4)

        # TODO [hyomin - Not just output, specific metric return required later]
        return out
