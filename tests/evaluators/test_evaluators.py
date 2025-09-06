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
import os
import pytest
import torch

from unittest.mock import MagicMock, patch

from compressai_vision.evaluators import COCOEVal

def mock_coco_api(annotation_file):
    # Create a mock COCO API object
    coco_api = MagicMock()
    coco_api.anns = {1: {'image_id': 1, 'category_id': 1, 'bbox': [10, 10, 50, 50]}}
    return coco_api

@pytest.fixture
def mock_dataset():
    dataset = MagicMock()
    dataset.dataset_name = "mock_coco_dataset"
    dataset.annotation_path = "mock_annotations.json"
    with open(dataset.annotation_path, 'w') as f:
        json.dump({
            'images': [{'id': 1, 'width': 640, 'height': 480}],
            'annotations': [{'id': 1, 'image_id': 1, 'category_id': 1, 'bbox': [10, 10, 50, 50]}],
            'categories': [{'id': 1, 'name': 'person'}]
        }, f)
    yield dataset
    os.remove(dataset.annotation_path)

@patch('detectron2.evaluation.COCOEvaluator', autospec=True)
@patch('compressai_vision.evaluators.evaluators.deccode_compressed_rle')
def test_coco_eval(mock_deccode, mock_evaluator, mock_dataset):
    # Configure the mock to have the expected attributes
    mock_coco_api_obj = MagicMock()
    mock_coco_api_obj.anns = {}
    mock_evaluator_instance = mock_evaluator.return_value
    mock_evaluator_instance._coco_api = mock_coco_api_obj
    mock_evaluator_instance.evaluate.return_value = {"AP": 50.0}

    evaluator = COCOEVal(
        datacatalog_name="MPEGOIV6",
        dataset_name="mock_coco_dataset",
        dataset=mock_dataset,
        output_dir="./test_output"
    )

    gt = [{'image_id': 1, 'width': 640, 'height': 480, 'image': torch.zeros(3, 480, 640)}]
    pred = [{'instances': MagicMock()}]

    evaluator.digest(gt, pred)
    mock_evaluator.return_value.process.assert_called_once_with(gt, pred)

    evaluator.results()
    mock_evaluator.return_value.evaluate.assert_called_once()
