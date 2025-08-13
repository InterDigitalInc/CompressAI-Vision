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
import math
import os

from collections import defaultdict
from pathlib import Path
from typing import Optional

import cv2
import motmetrics as mm
import numpy as np
import pandas as pd
import torch

from detectron2.data import MetadataCatalog
from detectron2.evaluation import COCOEvaluator
from detectron2.utils.visualizer import Visualizer
from jde.utils.io import unzip_objs
from mmpose.datasets.datasets import BaseCocoStyleDataset
from mmpose.datasets.transforms import PackPoseInputs
from mmpose.evaluation.metrics import CocoMetric
from pycocotools.coco import COCO
from pytorch_msssim import ms_ssim
from tqdm import tqdm
from yolox.data.datasets.coco import remove_useless_info
from yolox.evaluators import COCOEvaluator as YOLOX_COCOEvaluator
from yolox.utils import xyxy2xywh

from compressai_vision.datasets import deccode_compressed_rle
from compressai_vision.registry import register_evaluator
from compressai_vision.utils import time_measure, to_cpu

from .base_evaluator import BaseEvaluator
from .config import (
    coco_stuff_to_pandaset,
    coco_things_to_pandaset,
    pandaset_to_vcm_category,
    vcm_eval_category,
)
from .tf_evaluation_utils import (
    DetectionResultFields,
    InputDataFields,
    OpenImagesChallengeEvaluator,
    decode_gt_raw_data_into_masks_and_boxes,
    decode_masks,
    encode_masks,
)


# Function to calculate mIoU
def calculate_mIoU(gt, pred):
    def calculate_iou_for_class(gt, pred, class_id):
        # Calculate the intersection (true positives)
        true_positive = ((gt == class_id) & (pred == class_id)).sum()

        # Calculate the false positives and false negatives
        false_positive = ((gt != class_id) & (pred == class_id)).sum()
        false_negative = ((gt == class_id) & (pred != class_id)).sum()

        # Calculate the union
        union = true_positive + false_positive + false_negative

        # Return the IoU for the class
        return true_positive / union * 100 if union != 0 else 0

    iou_dict = {}
    total_iou = 0
    num_classes_in_data = 0

    for class_id in vcm_eval_category:
        # Check if the class is present in either ground truth or predictions
        # if np.any(gt == class_id) or np.any(pred == class_id):
        if np.any(gt == class_id):
            iou = calculate_iou_for_class(gt, pred, class_id)
            iou_dict[class_id] = iou
            total_iou += iou
            num_classes_in_data += 1
        else:
            iou_dict[class_id] = (
                np.nan
            )  # class not present in either ground truth or predictions

    # Calculate mean IoU
    mIoU = total_iou / num_classes_in_data if num_classes_in_data > 0 else 0

    return mIoU, iou_dict


@register_evaluator("COCO-EVAL")
class COCOEVal(BaseEvaluator):
    def __init__(
        self,
        datacatalog_name,
        dataset_name,
        dataset,
        output_dir="./vision_output/",
        eval_criteria="AP",
        **args,
    ):
        super().__init__(
            datacatalog_name, dataset_name, dataset, output_dir, eval_criteria
        )

        self.set_annotation_info(dataset)

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

    def save_visualization(self, gt, pred, output_dir, threshold):
        gt_image = gt[0]["image"]
        if torch.is_floating_point(gt_image):
            gt_image = (gt_image * 255).clamp(0, 255).to(torch.uint8)
            gt_image = gt_image[[2, 1, 0], ...]
        gt_image = gt_image.permute(1, 2, 0).cpu().numpy()
        gt_image = cv2.resize(gt_image, (gt[0]["width"], gt[0]["height"]))

        img_id = gt[0]["image_id"]
        metadata = MetadataCatalog.get(self.dataset_name)
        instances = pred[0]["instances"].to("cpu")
        if threshold:
            keep = instances.scores >= threshold
            instances = instances[keep]

        v = Visualizer(gt_image[:, :, ::-1], metadata, scale=1)
        out = v.draw_instance_predictions(
            instances
        )  # selected_instances for specific class
        output_path = os.path.join(output_dir, f"{img_id}.jpg")
        cv2.imwrite(output_path, out.get_image()[:, :, ::-1])
        return

    def results(self, save_path: str = None):
        out = self._evaluator.evaluate()

        if save_path:
            self.write_results(out, save_path)

        self.write_results(out)

        # summary = {}
        # for key, item_dict in out.items():
        #     summary[f"{key}"] = item_dict["AP"]

        return out


@register_evaluator("OIC-EVAL")
class OpenImagesChallengeEval(BaseEvaluator):
    def __init__(
        self,
        datacatalog_name,
        dataset_name,
        dataset,
        output_dir="./vision_output/",
        eval_criteria="AP50",
        **args,
    ):
        super().__init__(
            datacatalog_name, dataset_name, dataset, output_dir, eval_criteria
        )

        self.set_annotation_info(dataset)

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

    def save_visualization(self, gt, pred, output_dir, threshold):
        gt_image = gt[0]["image"]
        if torch.is_floating_point(gt_image):
            gt_image = (gt_image * 255).clamp(0, 255).to(torch.uint8)
            gt_image = gt_image[[2, 1, 0], ...]
        gt_image = gt_image.permute(1, 2, 0).cpu().numpy()
        gt_image = cv2.resize(gt_image, (gt[0]["width"], gt[0]["height"]))

        img_id = gt[0]["image_id"]
        metadata = MetadataCatalog.get(self.dataset_name)
        instances = pred[0]["instances"].to("cpu")

        if threshold:
            keep = instances.scores >= threshold
            instances = instances[keep]

        v = Visualizer(gt_image[:, :, ::-1], metadata, scale=1)
        out = v.draw_instance_predictions(
            instances
        )  # selected_instances for specific class
        output_path = os.path.join(output_dir, f"{img_id}.jpg")
        cv2.imwrite(output_path, out.get_image()[:, :, ::-1])
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

        start = time_measure()
        for pred_dict in self._predictions:
            img_id = pred_dict["img_id"]
            processed_dict = self._process_prediction(pred_dict)
            self._oic_evaluator.add_single_detected_image_info(img_id, processed_dict)

        self._logger.info(
            f"Elapsed time to process and register the predicted items to the evaluator: {time_measure() - start:.02f} sec"
        )
        out = self._oic_evaluator.evaluate()
        self._logger.info(f"Total evaluation time: {time_measure() - start:.02f} sec")

        if save_path:
            self.write_results(out, save_path)

        self.write_results(out)

        summary = {}
        for key, value in out.items():
            name = "-".join(key.split("/")[1:])
            summary[name] = value

        return summary


@register_evaluator("SEMANTICSEG-EVAL")
class SemanticSegmentationEval(BaseEvaluator):
    """
    Semantic Segmentation Evaluator

    Based on code from the Nokia Pandaset evaluation scripts
    """

    def __init__(
        self,
        datacatalog_name,
        dataset_name,
        dataset,
        output_dir="./vision_output/",
        eval_criteria="mIoU",
        **args,
    ):
        super().__init__(
            datacatalog_name, dataset_name, dataset, output_dir, eval_criteria
        )

        self._sem_gt = np.load(dataset.annotation_path, allow_pickle=True)["gt"]
        self.reset()

    def reset(self):
        # self._predictions = []
        # otuput sequence categories
        self._seq_gt_cats = []
        self._seq_det_cats = []
        self._frame_ctr = 0

    def digest(self, gt, pred):
        # Detectron segmentation
        seg_detectron = pred[0]["panoptic_seg"]
        panoptic_seg, segments_info = seg_detectron

        # frame ground truth, columns: x, y, cls
        frame_gt = self._sem_gt[self._frame_ctr]
        flat_indices = np.ravel_multi_index(
            (frame_gt[:, 1], frame_gt[:, 0]), (1080, 1920)
        )

        # get groundtruth categories
        gt_cats = frame_gt[:, 2]

        # convert detection from coco category to pandaset category
        det_ids = panoptic_seg.cpu().numpy().ravel()[flat_indices]

        # convert detectron2 id to pandaset categories
        cvt_table = np.zeros(80, dtype=int)
        for info in segments_info:
            if info["isthing"]:
                cvt_table[info["id"]] = coco_things_to_pandaset[info["category_id"]]
            else:
                cvt_table[info["id"]] = coco_stuff_to_pandaset[info["category_id"]]
        det_cats = cvt_table[det_ids]

        # convert to vcm categories
        vcm_cvt_table = np.zeros(50, dtype=int)
        for k, v in pandaset_to_vcm_category.items():
            vcm_cvt_table[k] = v

        gt_cats = vcm_cvt_table[gt_cats]
        det_cats = vcm_cvt_table[det_cats]

        # add frame results to sequence
        self._seq_gt_cats.append(gt_cats)
        self._seq_det_cats.append(det_cats)

        self._frame_ctr += 1

    def mIoU_eval(self):
        # Concatenate frame results to sequence results
        seq_gt = np.concatenate(self._seq_gt_cats)
        seq_det = np.concatenate(self._seq_det_cats)

        # Calculate mPA
        # mPA, class_mPA = calculate_mPA(seq_gt, seq_det)
        mIoU, class_mIoU = calculate_mIoU(seq_gt, seq_det)
        return mIoU, class_mIoU

    def results(self, save_path: str = None):
        mIoU, class_mIoU = self.mIoU_eval()
        class_mIoU["mIoU"] = mIoU

        if save_path:
            self.write_results(class_mIoU, save_path)

        self.write_results(class_mIoU)

        return class_mIoU


@register_evaluator("MOT-JDE-EVAL")
class MOT_JDE_Eval(BaseEvaluator):
    """
    A Multiple Object Tracking Evaluator

    This class evaluates MOT performance of tracking model such as JDE in compressai-vision.
    BaseEvaluator is inherited to interface with pipeline architecture in compressai-vision

    Functions below in this class refers to
        The class Evaluator inin Towards-Realtime-MOT/utils/evaluation.py at
        <https://github.com/Zhongdao/Towards-Realtime-MOT/blob/master/utils/evaluation.py>
        <https://github.com/Zhongdao/Towards-Realtime-MOT/blob/master/track.py>

        Full license statement can be found at
        <https://github.com/Zhongdao/Towards-Realtime-MOT/blob/master/LICENSE>

    """

    def __init__(
        self,
        datacatalog_name,
        dataset_name,
        dataset,
        output_dir="./vision_output/",
        eval_criteria="MOTA",
        apply_pred_offset=False,
        **args,
    ):
        super().__init__(
            datacatalog_name, dataset_name, dataset, output_dir, eval_criteria
        )
        self.apply_pred_offset = apply_pred_offset

        self.set_annotation_info(dataset)

        mm.lap.default_solver = "lap"
        self.dataset = dataset.dataset
        self.eval_info_file_name = self.get_jde_eval_info_name(self.dataset_name)

        self.reset()

    def reset(self):
        self.acc = mm.MOTAccumulator(auto_id=True)
        self._predictions = {}

    @staticmethod
    def _load_gt_in_motchallenge(filepath, fmt="mot15-2D", min_confidence=-1):
        return mm.io.loadtxt(filepath, fmt=fmt, min_confidence=min_confidence)

    @staticmethod
    def _format_pd_in_motchallenge(predictions: dict):
        all_indexes = []
        all_columns = []

        for frmID, preds in predictions.items():
            if len(preds) > 0:
                for data in preds:
                    tlwh, objID, conf = data

                    all_indexes.append((frmID, objID))

                    column_data = tlwh + (conf, -1, -1)
                    all_columns.append(column_data)

        columns = ["X", "Y", "Width", "Height", "Confidence", "ClassId", "Visibility"]

        idx = pd.MultiIndex.from_tuples(all_indexes, names=["FrameId", "Id"])
        return pd.DataFrame(all_columns, idx, columns)

    def digest(self, gt, pred):
        pred_list = []
        for tlwh, id in zip(pred["tlwhs"], pred["ids"]):
            x1, y1, w, h = tlwh
            if self.apply_pred_offset: # Replicate offset applied in load_motchallenge() in motmetrics library, used in VCM eval framework to load predictions from disk
                x1 -= 1
                y1 -= 1
            # x2, y2 = x1 + w, y1 + h
            parsed_pred = ((x1, y1, w, h), id, 1.0)
            pred_list.append(parsed_pred)

        self._predictions[int(gt[0]["image_id"])] = pred_list

    def save_visualization(self, gt, pred, output_dir, threshold):
        image_id = gt[0]["image_id"]
        gt_image = gt[0]["image"].permute(1, 2, 0).cpu().numpy()
        gt_image = (gt_image * 255).astype(np.uint8)
        gt_image = cv2.resize(
            cv2.cvtColor(gt_image, cv2.COLOR_RGB2BGR), (gt[0]["width"], gt[0]["height"])
        )
        online_im = self.plot_tracking(
            gt_image, pred["tlwhs"], pred["ids"], frame_id=image_id
        )
        output_path = os.path.join(output_dir, f"{image_id}.png")
        cv2.imwrite(output_path, online_im)
        return

    def plot_tracking(
        self, image, tlwhs, obj_ids, scores=None, frame_id=0, fps=0.0, ids2=None
    ):
        im = np.ascontiguousarray(np.copy(image))
        im_h, im_w = im.shape[:2]

        text_scale = max(1, image.shape[1] / 1600.0)
        text_thickness = 1 if text_scale > 1.1 else 1
        line_thickness = max(1, int(image.shape[1] / 500.0))

        for i, tlwh in enumerate(tlwhs):
            x1, y1, w, h = tlwh
            intbox = tuple(map(int, (x1, y1, x1 + w, y1 + h)))
            obj_id = int(obj_ids[i])
            id_text = "{}".format(int(obj_id))
            if ids2 is not None:
                id_text = id_text + ", {}".format(int(ids2[i]))
            color = self.get_color(abs(obj_id))
            cv2.rectangle(
                im, intbox[0:2], intbox[2:4], color=color, thickness=line_thickness
            )
            cv2.putText(
                im,
                id_text,
                (intbox[0], intbox[1] + 30),
                cv2.FONT_HERSHEY_PLAIN,
                text_scale,
                (0, 0, 255),
                thickness=text_thickness,
            )
        return im

    def get_color(self, idx):
        idx = idx * 3
        color = ((37 * idx) % 255, (17 * idx) % 255, (29 * idx) % 255)
        return color

    def results(self, save_path: str = None):
        out = self.mot_eval()

        if save_path:
            self.write_results(out, save_path)

        self.write_results(out)

        return out

    @staticmethod
    def digest_summary(summary):
        ret = {}
        keys_lists = [
            (
                [
                    "idf1",
                    "idp",
                    "idr",
                    "recall",
                    "precision",
                    "num_unique_objects",
                    "mota",
                    "motp",
                ],
                None,
            ),
            (
                [
                    "mostly_tracked",
                    "partially_tracked",
                    "mostly_lost",
                    "num_false_positives",
                    "num_misses",
                    "num_switches",
                    "num_fragmentations",
                    "num_transfer",
                    "num_ascend",
                    "num_migrate",
                ],
                int,
            ),
        ]

        for keys, dtype in keys_lists:
            selected_keys = [key for key in keys if key in summary]
            for key in selected_keys:
                ret[key] = summary[key] if dtype is None else dtype(summary[key])

        return ret

    def mot_eval(self):
        assert len(self.dataset) == len(
            self._predictions
        ), "Total number of frames are mismatch"

        # skip the very first frame
        for gt_frame in self.dataset[1:]:
            frm_id = int(gt_frame["image_id"])

            pred_objs = self._predictions[frm_id].copy()
            pred_tlwhs, pred_ids, _ = unzip_objs(pred_objs)

            gt_objs = gt_frame["annotations"]["gt"].copy()
            gt_tlwhs, gt_ids, _ = unzip_objs(gt_objs)

            gt_ignore = gt_frame["annotations"]["gt_ignore"].copy()
            gt_ignore_tlwhs, _, _ = unzip_objs(gt_ignore)

            # remove ignored results
            keep = np.ones(len(pred_tlwhs), dtype=bool)
            iou_distance = mm.distances.iou_matrix(
                gt_ignore_tlwhs, pred_tlwhs, max_iou=0.5
            )
            if len(iou_distance) > 0:
                match_is, match_js = mm.lap.linear_sum_assignment(iou_distance)
                match_is, match_js = map(
                    lambda a: np.asarray(a, dtype=int), [match_is, match_js]
                )
                match_ious = iou_distance[match_is, match_js]

                match_js = np.asarray(match_js, dtype=int)
                match_js = match_js[np.logical_not(np.isnan(match_ious))]
                keep[match_js] = False
                pred_tlwhs = pred_tlwhs[keep]
                pred_ids = pred_ids[keep]

            # get distance matrix
            iou_distance = mm.distances.iou_matrix(gt_tlwhs, pred_tlwhs, max_iou=0.5)

            # accumulate
            self.acc.update(gt_ids, pred_ids, iou_distance)

        # get summary
        metrics = mm.metrics.motchallenge_metrics
        mh = mm.metrics.create()

        summary = mh.compute(
            self.acc,
            metrics=metrics,
            name=self.dataset_name,
            return_dataframe=False,
            return_cached=True,
        )

        return self.digest_summary(summary)

    def _save_all_eval_info(self, pred: dict):
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)
        file_name = f"{self.output_dir}/{self.eval_info_file_name}"
        torch.save(pred, file_name)

    def _load_all_eval_info(self):
        file_name = f"{self.output_dir}/{self.eval_info_file_name}"
        return torch.load(file_name)


@register_evaluator("MOT-TVD-EVAL")
class MOT_TVD_Eval(MOT_JDE_Eval):
    """
    A Multiple Object Tracking Evaluator for TVD

    This class evaluates MOT performance of tracking model such as JDE specifically on TVD
    """

    def __init__(
        self,
        datacatalog_name,
        dataset_name,
        dataset,
        output_dir="./vision_output/",
        eval_criteria="MOTA",
        **args,
    ):
        super().__init__(
            datacatalog_name, dataset_name, dataset, output_dir, eval_criteria, **args
        )

        self.set_annotation_info(dataset)

        self._gt_pd = self._load_gt_in_motchallenge(self.annotation_path)

        assert self.seqinfo_path is not None, "Sequence Information must be provided"

    def mot_eval(self):
        assert len(self.dataset) == len(
            self._predictions
        ), "Total number of frames are mismatch"

        self._save_all_eval_info(self._predictions)
        _pd_pd = self._format_pd_in_motchallenge(self._predictions)

        acc, ana = mm.utils.CLEAR_MOT_M(self._gt_pd, _pd_pd, self.seqinfo_path)

        # get summary
        metrics = mm.metrics.motchallenge_metrics
        mh = mm.metrics.create()

        summary = mh.compute(
            acc,
            ana=ana,
            metrics=metrics,
            name=self.dataset_name,
            return_dataframe=False,
            return_cached=True,
        )

        return self.digest_summary(summary)


@register_evaluator("MOT-HIEVE-EVAL")
class MOT_HiEve_Eval(MOT_JDE_Eval):
    """
    A Multiple Object Tracking Evaluator for HiEve

    This class evaluates MOT performance of tracking model such as JDE specifically on HiEve

    """

    def __init__(
        self,
        datacatalog_name,
        dataset_name,
        dataset,
        output_dir="./vision_output/",
        eval_criteria="MOTA",
        **args,
    ):
        super().__init__(
            datacatalog_name, dataset_name, dataset, output_dir, eval_criteria, **args
        )

        self.set_annotation_info(dataset)

        mm.lap.default_solver = "munkres"

        self._gt_pd = self._load_gt_in_motchallenge(
            self.annotation_path, min_confidence=1
        )

    def mot_eval(self):
        assert len(self.dataset) == len(
            self._predictions
        ), "Total number of frames are mismatch"

        self._save_all_eval_info(self._predictions)
        _pd_pd = self._format_pd_in_motchallenge(self._predictions)

        acc = mm.utils.compare_to_groundtruth(self._gt_pd, _pd_pd)

        # get summary
        metrics = mm.metrics.motchallenge_metrics
        mh = mm.metrics.create()

        summary = mh.compute(
            acc,
            metrics=metrics,
            name=self.dataset_name,
            return_dataframe=False,
            return_cached=True,
        )

        return self.digest_summary(summary)


@register_evaluator("YOLOX-COCO-EVAL")
class YOLOXCOCOEval(BaseEvaluator):
    def __init__(
        self,
        datacatalog_name,
        dataset_name,
        dataset,
        output_dir="./vision_output/",
        eval_criteria="AP",
        **args,
    ):
        super().__init__(
            datacatalog_name, dataset_name, dataset, output_dir, eval_criteria
        )

        self.set_annotation_info(dataset)

        cocoapi = COCO(self.annotation_path)
        remove_useless_info(cocoapi)
        class_ids = sorted(cocoapi.getCatIds())
        cats = cocoapi.loadCats(cocoapi.getCatIds())

        class dummy_dataloader:
            def __init__(self):
                class dummy_dataset:
                    def __init__(self):
                        self.coco = cocoapi
                        self.class_ids = class_ids
                        self.cats = cats

                self.dataset = dummy_dataset()
                self.batch_size = 1

        dataloader = dummy_dataloader()

        self._class_ids = class_ids
        self._img_size = dataset.input_size
        self._evaluator = YOLOX_COCOEvaluator(dataloader, None, -1, -1, -1)
        self.reset()

    def reset(self):
        self.data_list = []
        self.output_data = defaultdict()

    def digest(self, gt, pred):
        assert len(gt) == 1

        img_heights = [gt[0]["height"]]
        img_widths = [gt[0]["width"]]
        img_ids = [gt[0]["image_id"]]

        data_list_elem, image_wise_data = self._convert_to_coco_format(
            pred, [img_heights, img_widths], img_ids
        )
        self.data_list.extend(data_list_elem)
        self.output_data.update(image_wise_data)

    def results(self, save_path: str = None):
        dummy_statistics = torch.FloatTensor([0, 0, len(self.output_data)])
        eval_results = self._evaluator.evaluate_prediction(
            self.data_list, dummy_statistics
        )

        assert self.output_dir is not None

        file_path = Path(f"{self.output_dir}")
        if not file_path.is_dir():
            self._logger.info(f"creating output folder: {file_path}")
            file_path.mkdir(parents=True, exist_ok=True)
            self._logger.info("Saving results to {}".format(file_path))

        with open(f"{file_path}/{self.get_coco_eval_info_name()}", "w") as f:
            f.write(json.dumps(self.data_list))
            f.flush()

        if save_path:
            self.write_results(eval_results, save_path)

        self.write_results(eval_results)

        *listed_items, summary = eval_results

        self._logger.info("\n" + summary)

        return {"AP": listed_items[0] * 100, "AP50": listed_items[1] * 100}

    def _convert_to_coco_format(self, outputs, info_imgs, ids):
        # reference : yolox > evaluators > coco_evaluator > convert_to_coco_format
        data_list = []
        image_wise_data = defaultdict(dict)
        for output, img_h, img_w, img_id in zip(
            outputs, info_imgs[0], info_imgs[1], ids
        ):
            if output is None:
                continue
            output = output.cpu()

            bboxes = output[:, 0:4]

            # preprocessing: resize
            scale = min(
                self._img_size[0] / float(img_h), self._img_size[1] / float(img_w)
            )
            bboxes /= scale
            cls = output[:, 6]
            scores = output[:, 4] * output[:, 5]

            image_wise_data.update(
                {
                    img_id: {
                        "bboxes": [box.numpy().tolist() for box in bboxes],
                        "scores": [score.numpy().item() for score in scores],
                        "categories": [
                            self._class_ids[int(cls[ind])]
                            for ind in range(bboxes.shape[0])
                        ],
                    }
                }
            )

            bboxes = xyxy2xywh(bboxes)

            for ind in range(bboxes.shape[0]):
                label = self._class_ids[int(cls[ind])]
                pred_data = {
                    "image_id": img_id,
                    "category_id": label,
                    "bbox": bboxes[ind].numpy().tolist(),
                    "score": scores[ind].numpy().item(),
                    "segmentation": [],
                }  # COCO json format
                data_list.append(pred_data)

        return data_list, image_wise_data


@register_evaluator("MMPOSE-COCO-EVAL")
class MMPOSECOCOEval(BaseEvaluator):
    def __init__(
        self,
        datacatalog_name,
        dataset_name,
        dataset,
        output_dir="./vision_output/",
        eval_criteria="AP",
        **args,
    ):
        super().__init__(
            datacatalog_name, dataset_name, dataset, output_dir, eval_criteria
        )

        self.set_annotation_info(dataset)
        self.input_size = dataset.input_size
        self.comput_scale_and_center = (
            dataset.get_org_mapper_func().compute_scale_and_center
        )

        if "metainfo" in args:
            metainfo = args["metainfo"]
        else:
            self._logger.warning("No Meta Infomation provided. Set default values")
            import mmpose

            default_config_path = Path(
                f"{mmpose.__path__[0]}/../configs/_base_/datasets/coco.py"
            )
            default_config_path = default_config_path.resolve()
            assert default_config_path.exists() and default_config_path.is_file()
            metainfo = str(default_config_path)

        # currently only support bottomup case
        loaded_dataset = BaseCocoStyleDataset(
            ann_file=self.annotation_path,
            metainfo={"from_file": metainfo},
            data_mode="bottomup",
            test_mode=True,
            serialize_data=False,
        )

        _metainfo = loaded_dataset.metainfo
        _mmpose_coco_style_dataset = loaded_dataset.data_list
        assert len(dataset.dataset) == len(_mmpose_coco_style_dataset)
        self._loaded_data_sample_size = len(dataset.dataset)

        _meta_keys = (
            "id",
            "img_id",
            "img_path",
            "ori_shape",
            "img_shape",
            "input_size",
            "input_center",
            "input_scale",
        )
        convert_packposeinput = PackPoseInputs(_meta_keys)
        self._annotation_per_sample = defaultdict()
        dummy_img = np.random.rand(1, 1, 3)
        for data_sample in _mmpose_coco_style_dataset:
            img_id = data_sample["img_id"]
            data_sample["img"] = dummy_img
            out = convert_packposeinput(data_sample)
            self._annotation_per_sample[img_id] = out["data_samples"]
            out.clear()

        self._evaluator = CocoMetric(
            ann_file=self.annotation_path, score_mode="bbox", nms_mode="none"
        )
        self._evaluator.dataset_meta = _metainfo
        self.reset()

    def reset(self):
        self.pred_data_list_cnt = 0

    def digest(self, gts, preds):
        assert len(gts) == 1 and len(preds) == 1

        src_img_height = gts[0]["height"]
        src_img_width = gts[0]["width"]
        img_id = gts[0]["image_id"]
        pred = preds[0]

        # mmpose style scaling
        input_scale, input_center = self.comput_scale_and_center(
            src_img_width, src_img_height
        )

        # convert keypoint coordinates from input space to image space
        pred.keypoints = (
            (pred.keypoints / self.input_size) * input_scale
            + input_center
            - (0.5 * input_scale)
        )
        assert "keypoints_visible" in pred

        # convert bbox coordinates from input space to image space
        if "bboxes" in pred:
            bboxes = pred.bboxes.reshape(pred.bboxes.shape[0], 2, 2)
            bboxes = (
                (bboxes / self.input_size) * input_scale
                + input_center
                - (0.5 * input_scale)
            )
            pred.bboxes = bboxes.reshape(bboxes.shape[0], 4)

        _data_sample = self._annotation_per_sample[img_id]
        _data_sample.pred_instances = pred
        self._evaluator.process({"dummy": None}, [_data_sample.to_dict()])
        self.pred_data_list_cnt = self.pred_data_list_cnt + 1

    def results(self, save_path: str = None):
        assert self._loaded_data_sample_size == self.pred_data_list_cnt
        eval_results = self._evaluator.evaluate(self.pred_data_list_cnt)

        assert self.output_dir is not None

        file_path = Path(f"{self.output_dir}")
        if not file_path.is_dir():
            self._logger.info(f"creating output folder: {file_path}")
            file_path.mkdir(parents=True, exist_ok=True)
            self._logger.info("Saving results to {}".format(file_path))

        # Nothing to dump for classwise for the case
        # with open(f"{file_path}/{self.get_coco_eval_info_name()}", "w") as f:
        #    f.write(json.dumps(self.data_list))
        #    f.flush()

        if save_path:
            self.write_results(eval_results, save_path)

        self.write_results(eval_results)

        # item_keys = list(eval_results.keys())
        item_vals = list(eval_results.values())

        # self._logger.info("\n" + summary)

        return {"AP": item_vals[0] * 100, "AP50": item_vals[1] * 100}


@register_evaluator("VISUAL-QUALITY-EVAL")
class VisualQualityEval(BaseEvaluator):
    def __init__(
        self,
        datacatalog_name,
        dataset_name,
        dataset,
        output_dir="./vision_output/",
        eval_criteria="psnr",
        **args,
    ):
        super().__init__(
            datacatalog_name, dataset_name, dataset, output_dir, eval_criteria
        )

        self.reset()

    @staticmethod
    def compute_psnr(a, b):
        mse = torch.mean((a - b) ** 2).item()
        return -10 * math.log10(mse)

    @staticmethod
    def compute_msssim(a, b):
        return ms_ssim(a, b, data_range=1.0).item()

    def reset(self):
        self._evaluations = []
        self._sum_psnr = 0
        self._sum_msssim = 0
        self._cc = 0

    def write_results(self, path: str = None):
        if path is None:
            path = f"{self.output_dir}"

        path = Path(path)
        if not path.is_dir():
            self._logger.info(f"creating output folder: {path}")
            path.mkdir(parents=True, exist_ok=True)

        with open(f"{path}/{self.output_file_name}.json", "w", encoding="utf-8") as f:
            json.dump(self._evaluations, f, ensure_ascii=False, indent=4)

    def digest(self, gt, pred):
        ref = gt[0]["image"].unsqueeze(0).cpu()
        tst = pred.unsqueeze(0).cpu()

        assert ref.shape == tst.shape

        psnr = self.compute_psnr(ref, tst)
        msssim = self.compute_msssim(ref, tst)

        eval_dict = {
            "img_id": gt[0]["image_id"],
            "img_size": (gt[0]["height"], gt[0]["width"]),
            "msssim": msssim,
            "psnr": psnr,
        }

        self._sum_psnr += psnr
        self._sum_msssim += msssim
        self._evaluations.append(eval_dict)

        self._cc += 1

    def results(self, save_path: str = None):
        if save_path:
            self.write_results(save_path)

        self.write_results()

        summary = {
            "msssim": (self._sum_msssim / self._cc),
            "psnr": (self._sum_psnr / self._cc),
        }
        return summary
