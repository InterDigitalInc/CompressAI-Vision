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


import json
import logging
from pathlib import Path

import torch.nn as nn


class BaseEvaluator(nn.Module):
    def __init__(
        self,
        datacatalog_name,
        dataset_name,
        dataset,
        output_dir="./vision_output/",
        criteria=None,
    ):
        self._logger = logging.getLogger(self.__class__.__name__)
        self.datacatalog_name = datacatalog_name
        self.dataset_name = dataset_name
        self.output_dir = output_dir
        self.criteria = criteria
        self.output_file_name = (
            f"{self.__class__.__name__}_on_{datacatalog_name}_{dataset_name}"
        )

        path = Path(self.output_dir)
        if (not path.is_dir()) and not path.exists():
            self._logger.info(f"creating output folder: {path}")
            path.mkdir(parents=True, exist_ok=True)

    def set_annotation_info(self, dataset):
        self.annotation_path = dataset.annotation_path
        self.seqinfo_path = dataset.seqinfo_path
        self.thing_classes = dataset.thing_classes
        self.thing_id_mapping = dataset.thing_dataset_id_to_contiguous_id

    @staticmethod
    def get_jde_eval_info_name(name):
        return f"{name}_info_to_eval.h5"

    @staticmethod
    def get_coco_eval_info_name(name):
        return "coco_instances_results.json"

    def reset(self):
        raise NotImplementedError

    def digest(self, gt, pred):
        raise NotImplementedError

    def results(self, save_path: str = None):
        raise NotImplementedError

    def write_results(self, out, path: str = None):
        if path is None:
            path = f"{self.output_dir}"

        path = Path(path)
        if not path.is_dir():
            self._logger.info(f"creating output folder: {path}")
            path.mkdir(parents=True, exist_ok=True)

        with open(f"{path}/{self.output_file_name}.json", "w", encoding="utf-8") as f:
            json.dump(out, f, ensure_ascii=False, indent=4)
