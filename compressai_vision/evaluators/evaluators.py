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

import copy
import json
import os
import time

import motmetrics as mm
import numpy as np
import torch
from detectron2.evaluation import COCOEvaluator
from jde.utils.io import unzip_objs
from tqdm import tqdm

from compressai_vision.datasets import deccode_compressed_rle
from compressai_vision.registry import register_evaluator
from compressai_vision.utils import to_cpu

from .base_evaluator import BaseEvaluator
from .tf_evaluation_utils import (
    DetectionResultFields,
    InputDataFields,
    OpenImagesChallengeEvaluator,
    decode_gt_raw_data_into_masks_and_boxes,
    decode_masks,
    encode_masks,
)


@register_evaluator("COCO-EVAL")
class COCOEVal(BaseEvaluator):
    def __init__(
        self, datacatalog_name, dataset_name, dataset, output_dir="./vision_output/"
    ):
        super().__init__(datacatalog_name, dataset_name, dataset, output_dir)

        self._evaluator = COCOEvaluator(
            dataset_name, False, output_dir=output_dir, use_fast_impl=False
        )

        if datacatalog_name == "MPEGOIV6":
            deccode_compressed_rle(self._evaluator._coco_api.anns)

        self.reset()

    def reset(self):
        self._evaluator.reset()

    def digest(self, gt, pred):
        return self._evaluator.process(gt, pred)

    def results(self, save_path: str = None):
        out = self._evaluator.evaluate()

        if save_path:
            self.write_results(out, save_path)

        self.write_results(out)

        summary = {}
        for key, item_dict in out.items():
            summary.update({f"{key}": item_dict["AP"]})

        return summary


@register_evaluator("OIC-EVAL")
class OpenImagesChallengeEval(BaseEvaluator):
    def __init__(
        self, datacatalog_name, dataset_name, dataset, output_dir="./vision_output/"
    ):
        super().__init__(datacatalog_name, dataset_name, dataset, output_dir)

        with open(dataset.annotation_path) as f:
            json_dict = json.load(f)

        def _search_category(name):
            for e, item in enumerate(self.thing_classes):
                if name == self._normalize_labelname(item):
                    return e

            self._logger.error(f"Not found Item {name} in 'thing_classes'")
            raise ValueError

        assert len(json_dict["annotations"]) > 0
        has_segmentation = (
            True if "segmentation" in json_dict["annotations"][0] else False
        )

        self._valid_contiguous_id = []
        self._oic_labelmap_dict = {}
        self._oic_categories = []

        assert "oic_labelmap" in json_dict
        assert "oic_annotations" in json_dict

        valid_categories = json_dict["oic_labelmap"]
        gt_annotations = json_dict["oic_annotations"]

        for item in valid_categories:
            e = _search_category(self._normalize_labelname(item["name"]))

            if e is not None:
                self._valid_contiguous_id.append(e)

            self._oic_labelmap_dict[self._normalize_labelname(item["name"])] = item[
                "id"
            ]
            self._oic_categories.append({"id": item["id"], "name": item["name"]})

        self._oic_evaluator = OpenImagesChallengeEvaluator(
            self._oic_categories, evaluate_masks=has_segmentation
        )

        self._logger.info(
            f"Loading annotations for {len(gt_annotations)} images and reformatting them for OpenImageChallenge Evaluator."
        )
        for gt in tqdm(gt_annotations):
            img_id = gt["image_id"]
            annotations = gt["annotations"]

            anno_dict = {}
            anno_dict[InputDataFields.groundtruth_boxes] = np.array(annotations["bbox"])
            anno_dict[InputDataFields.groundtruth_classes] = np.array(
                annotations["cls"]
            )
            anno_dict[InputDataFields.groundtruth_group_of] = np.array(
                annotations["group_of"]
            )

            img_cls = []
            for name in annotations["img_cls"]:
                img_cls.append(self._oic_labelmap_dict[self._normalize_labelname(name)])
            anno_dict[InputDataFields.groundtruth_image_classes] = np.array(img_cls)

            if "mask" in annotations:
                segments, _ = decode_gt_raw_data_into_masks_and_boxes(
                    annotations["mask"], annotations["img_size"]
                )
                anno_dict[InputDataFields.groundtruth_instance_masks] = segments

            self._oic_evaluator.add_single_ground_truth_image_info(img_id, anno_dict)

        self._logger.info(
            f"All groundtruth annotations for {len(gt_annotations)} images are successfully registred to the evaluator."
        )

        self.reset()

    def reset(self):
        self._predictions = []
        self._cc = 0

    @staticmethod
    def _normalize_labelname(name: str):
        return name.lower().replace(" ", "_")

    def digest(self, gt, pred):
        assert len(gt) == len(pred) == 1, "Batch size must be 1 for the evaluation"

        if self._oic_evaluator is None:
            self._logger.warning(
                "There is no assigned evaluator for the class. Evaluator will not work properly"
            )
            return

        img_id = gt[0]["image_id"]
        test_fields = to_cpu(pred[0]["instances"])

        imgH, imgW = test_fields.image_size

        classes = test_fields.pred_classes
        scores = test_fields.scores
        bboxes = test_fields.pred_boxes.tensor

        assert len(classes) == len(scores) == len(bboxes)

        masks = []
        has_mask = test_fields.has("pred_masks")
        if has_mask:
            masks = test_fields.pred_masks
            assert len(masks) == len(bboxes)

        valid_indexes = []
        for e, cid in enumerate(classes):
            if cid in self._valid_contiguous_id:
                valid_indexes.append(e)

        if len(valid_indexes) > 0:
            pred_dict = {
                "img_id": img_id,
                "img_size": (imgH, imgW),
                "classes": classes[valid_indexes].tolist(),
                "scores": scores[valid_indexes].tolist(),
                "bboxes": bboxes[valid_indexes].tolist(),
            }
            if has_mask:
                pred_dict["masks"] = encode_masks(masks[valid_indexes])

            self._predictions.append(pred_dict)

        self._cc += 1

        return

    def _process_prediction(self, pred_dict):
        valid_cls = []
        valid_scores = []
        valid_bboxes = []
        valid_segment_masks = []
        valid_segment_boxes = []

        imgH, imgW = pred_dict["img_size"]
        classes = pred_dict["classes"]
        scores = pred_dict["scores"]
        bboxes = pred_dict["bboxes"]

        has_mask = True if "masks" in pred_dict else False
        if has_mask:
            masks = pred_dict["masks"]

        for e, _items in enumerate(zip(classes, scores, bboxes)):
            _class, _score, _bbox = _items

            cate_name = self._normalize_labelname(self.thing_classes[_class])
            valid_cls.append(self._oic_labelmap_dict[cate_name])

            valid_scores.append(_score)
            norm_bbox = np.array(_bbox) / [imgW, imgH, imgW, imgH]
            # XMin, YMin, XMax, YMax --> YMin, XMin, YMax, XMax
            valid_bboxes.append(norm_bbox[[1, 0, 3, 2]])

            if has_mask:
                segment, boxe = decode_masks(masks[e])
                valid_segment_masks.append(segment)
                valid_segment_boxes.append(boxe)

        res_dict = {
            DetectionResultFields.detection_classes: np.array(valid_cls),
            DetectionResultFields.detection_scores: np.array(valid_scores).astype(
                float
            ),
        }

        if has_mask:
            res_dict[DetectionResultFields.detection_masks] = np.concatenate(
                valid_segment_masks, axis=0
            )
            res_dict[DetectionResultFields.detection_boxes] = np.concatenate(
                valid_segment_boxes, axis=0
            )
        else:
            res_dict[DetectionResultFields.detection_boxes] = np.array(
                valid_bboxes
            ).astype(float)

        return res_dict

    def results(self, save_path: str = None):
        if self._oic_evaluator is None:
            self._logger.warning(
                "There is no assigned evaluator for the class. Evaluator will not work properly"
            )
            return

        if len(self._predictions) == 0:
            self._logger.warning("There is no detected objects to evaluate")
            return

        start = time.time()
        for pred_dict in self._predictions:
            img_id = pred_dict["img_id"]
            processed_dict = self._process_prediction(pred_dict)
            self._oic_evaluator.add_single_detected_image_info(img_id, processed_dict)

        self._logger.info(
            f"Elapsed time to process and register the predicted items to the evaluator: {time.time() - start:.02f} sec"
        )
        out = self._oic_evaluator.evaluate()
        self._logger.info(f"Total evaluation time: {time.time() - start:.02f} sec")

        if save_path:
            self.write_results(out, save_path)

        self.write_results(out)

        summary = {}
        for key, value in out.items():
            name = "-".join(key.split("/")[1:])
            summary.update({name: value})

        return summary


@register_evaluator("MOT-EVAL")
class MOTEval(BaseEvaluator):
    """
    A Multiple Object Tracking Evaluator

    This class evaluates MOT performance of tracking model such as JDE in compressai-vision.
    BaseEvaluator is inherited to interface with pipeline architecture in compressai-vision

    Functions below in this class refers to
        The class Evaluator inin Towards-Realtime-MOT/utils/evaluation.py at
        <https://github.com/Zhongdao/Towards-Realtime-MOT/blob/master/utils/evaluation.py>

        Full license statement can be found at
        <https://github.com/Zhongdao/Towards-Realtime-MOT/blob/master/LICENSE>

    """

    def __init__(
        self, datacatalog_name, dataset_name, dataset, output_dir="./vision_output/"
    ):
        super().__init__(datacatalog_name, dataset_name, dataset, output_dir)

        mm.lap.default_solver = "lap"

        self.reset()

    def reset(self):
        self.acc = mm.MOTAccumulator(auto_id=True)
        self._predictions = {}

    def digest(self, gt, pred):
        print("Let us do something here")
        return None  # self._evaluator.process(gt, pred)

    def results(self, save_path: str = None):
        out = self._evaluator.evaluate()

        if save_path:
            self.write_results(out, save_path)

        self.write_results(out)

        summary = {}
        for key, item_dict in out.items():
            summary.update({f"{key}": item_dict["AP"]})

        return summary


@register_evaluator("YOLO-EVAL")
class YOLOEval(BaseEvaluator):
    def __init__(
        self, datacatalog_name, dataset_name, dataset, output_dir="./vision_output/"
    ):
        super().__init__(datacatalog_name, dataset_name, dataset, output_dir)
