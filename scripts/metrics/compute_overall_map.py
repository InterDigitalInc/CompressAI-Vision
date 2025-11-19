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

r"""
Compute overall mAP over some sequences outputs



"""

from __future__ import annotations

import argparse
import csv
import json
import os

from typing import Any, List

import numpy as np
import pandas as pd
import utils


# from detectron2.evaluation import COCOEvaluator
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from compressai_vision.evaluators.evaluators import BaseEvaluator

CLASSES = ["CLASS-AB", "CLASS-C", "CLASS-D", "CLASS-AB*"]

SEQS_BY_CLASS = {
    CLASSES[0]: [
        "Traffic_2560x1600_30",
        "Kimono_1920x1080_24",
        "ParkScene_1920x1080_24",
        "Cactus_1920x1080_50",
        "BasketballDrive_1920x1080_50",
        "BQTerrace_1920x1080_60",
    ],
    CLASSES[1]: ["BasketballDrill_832x480_50", "BQMall_832x480_60", "PartyScene_832x480_50", "RaceHorses_832x480_832x480_30"],
    CLASSES[2]: ["BasketballPass_416x240_50", "BQSquare_416x240_60", "BlowingBubbles_416x240_50", "RaceHorses_416x240_30"],
    CLASSES[3]: ["ns_Traffic_2560x1600_30", "ns_BQTerrace_1920x1080_60"],
}

SEQUENCE_TO_OFFSET = {
    "Traffic_2560x1600_30": 10000,
    "Kimono_1920x1080_24": 20000,
    "ParkScene_1920x1080_24": 30000,
    "Cactus_1920x1080_50": 40000,
    "BasketballDrive_1920x1080_50": 50000,
    "BQTerrace_1920x1080_60": 60000,
    "BasketballDrill_832x480_50": 70000,
    "BQMall_832x480_60": 80000,
    "PartyScene_832x480_50": 90000,
    "RaceHorses_832x480_30": 100000,
    "BasketballPass_416x240_50": 110000,
    "BQSquare_416x240_60": 120000,
    "BlowingBubbles_416x240_50": 130000,
    "RaceHorses_416x240_30": 140000,
}

TMP_EVAL_FILE = "tmp_eval.json"
TMP_ANCH_FILE = "tmp_anch.json"

NS_SEQ_PREFIX = "ns_" # Prefix of non-scaled sequences

def compute_overall_mAP(seq_root_names, items):

    classwise_instances_results = []
    classwise_anchor_images = []
    classwise_annotation = []
    categories = None
    annotation_id = 0
    for e, (item, root_name) in enumerate(zip(items, seq_root_names)):
        assert root_name in item[utils.SEQ_NAME_KEY], f"Not found {root_name} in {item[utils.SEQ_NAME_KEY]} {utils.SEQ_NAME_KEY}"

        root_name = root_name.replace(NS_SEQ_PREFIX, "")

        seq_img_id_offset = SEQUENCE_TO_OFFSET[root_name]

        with open(item[utils.EVAL_INFO_KEY], "r") as f:
            eval_data = json.load(f)

        for d in eval_data:
            d["image_id"] = int(d["image_id"]) + seq_img_id_offset
            classwise_instances_results.append(d)

        with open(item[utils.GT_INFO_KEY], "r") as f:
            gt_data = json.load(f)

        # images
        for d in gt_data["images"]:
            d["id"] = d["id"] + seq_img_id_offset
            classwise_anchor_images.append(d)

        for d in gt_data["annotations"]:
            annotation_id = annotation_id + 1

            d["id"] = d["id"] = annotation_id
            d["image_id"] = d["image_id"] + seq_img_id_offset
            classwise_annotation.append(d)

        if e == 0:
            categories = gt_data["categories"]

    classwise_gt_data = {
        "images": classwise_anchor_images,
        "categories": categories,
        "annotations": classwise_annotation,
    }

    with open(TMP_EVAL_FILE, "w") as f:
        json.dump(classwise_instances_results, f, indent=4)

    with open(TMP_ANCH_FILE, "w") as f:
        json.dump(classwise_gt_data, f, indent=4)

    summary = coco_evaluation(TMP_ANCH_FILE, TMP_EVAL_FILE)

    os.remove(TMP_EVAL_FILE)
    os.remove(TMP_ANCH_FILE)

    # print("\n")
    # print(summary)
    # print("\n")

    return summary


# this function is originated from MPEG FCVCM Anchor s/w package (i.e., mpeg-fcvcm-sfu-objdet-anchor)
def coco_evaluation(ann_file, detections):
    if not os.path.isfile(detections):
        return None
    coco = COCO(ann_file)
    coco_res = coco.loadRes(
        detections
    )  # JSON file like <sim_dir>/<test_name>/coco_instances_results.json
    coco_eval = COCOeval(coco, coco_res, "bbox")
    coco_eval.params.imgIds = coco.getImgIds()  # image IDs to evaluate
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    import logging

    class dummyclass:
        def __init__(self):
            self._logger = logging.getLogger(__name__)

    # things = [i["name"] for i in coco_eval.cocoGt.cats.values()]
    # out_all = COCOEvaluator._derive_coco_results(
    #     dummyclass(), coco_eval, iou_type="bbox", class_names=things
    # )

    headers = ["AP", "AP50", "AP75", "APS", "APM", "APL"]
    npstat = np.array(coco_eval.stats[:6])
    npstat = npstat * 100  # Percent
    # npstat = np.around(npstat, 2)
    data_frame = pd.DataFrame([npstat], columns=headers)

    return data_frame


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-r",
        "--result_path",
        required=True,
        help="For example, '.../logs/runs/[pipeline]/[codec]/[datacatalog]/' ",
    )
    parser.add_argument(
        "-q",
        "--quality_index",
        required=False,
        default=-1,
        type=int,
        help="Provide index of quality folders under the `result_path'. quality_index is only meant to point the orderd folders by qp names because there might be different range of qps are used for different sequences",
    )
    parser.add_argument(
        "-a",
        "--all_qualities",
        action="store_true",
        help="run all 6 rate points in MPEG CTCs",
    )
    parser.add_argument(
        "-d",
        "--dataset_path",
        required=True,
        help="For example, '.../vcm_testdata/[dataset]' ",
    )
    parser.add_argument(
        "-c",
        "--class_to_compute",
        type=str,
        choices=CLASSES,
        required=True,
    )

    args = parser.parse_args()
    if args.all_qualities:
        qualities = range(0, 6)
    else:
        qualities = [args.quality_index]

    with open(
        f"{args.result_path}/{args.class_to_compute}.csv", "w", newline=""
    ) as file:
        writer = csv.writer(file)
        for q in qualities:
            items = utils.search_items(
                args.result_path,
                args.dataset_path,
                q,
                SEQS_BY_CLASS[args.class_to_compute],
                BaseEvaluator.get_coco_eval_info_name,
            )

            assert (
                len(items) > 0
            ), "Nothing relevant information found from given directories..."

            summary = compute_overall_mAP(SEQS_BY_CLASS[args.class_to_compute], items)

            writer.writerow([f"{q}", f"{summary['AP'][0]:.4f}"])
            print(f"{'=' * 10} FINAL OVERALL mAP SUMMARY {'=' * 10}")
            print(f"{'-' * 32} AP : {summary['AP'][0]:.4f}")
            print("\n\n")
