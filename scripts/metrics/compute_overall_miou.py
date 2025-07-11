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
Compute overall MOT over some sequences outputs



"""

from __future__ import annotations

import argparse
import csv
import json

import utils

from compressai_vision.evaluators.evaluators import BaseEvaluator

from .compute_overall_mota import compute_overall_mota

CLASSES = ["PANDAM1", "PANDAM2", "PANDAM2"]

SEQS_BY_CLASS = {
    CLASSES[0]: [
        "PANDA057",
        "PANDA058",
        "PANDA069",
        "PANDA070",
        "PANDA072",
        "PANDA073",
        "PANDA077",
    ],
    CLASSES[1]: [
        "PANDA003",
        "PANDA011",
        "PANDA016",
        "PANDA017",
        "PANDA021",
        "PANDA023",
        "PANDA027",
        "PANDA029",
        "PANDA030",
        "PANDA033",
        "PANDA035",
        "PANDA037",
        "PANDA039",
        "PANDA043",
        "PANDA053",
        "PANDA056",
        "PANDA097",
    ],
    CLASSES[2]: [
        "PANDA088",
        "PANDA089",
        "PANDA090",
        "PANDA095",
        "PANDA109",
        "PANDA112",
        "PANDA113",
        "PANDA115",
        "PANDA117",
        "PANDA119",
        "PANDA122",
        "PANDA124",
    ],
}


def compute_overall_mIoU(class_name, items):
    miou_acc = 0.0
    for item in items:
        with open(item["eval_info"], "r") as f:
            results = json.load(f)
            miou_acc += results["mIoU"]

    miou_acc = miou_acc / len(items)

    return miou_acc


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
                BaseEvaluator.get_jde_eval_info_name,
            )

            assert (
                len(items) > 0
            ), "Nothing relevant information found from given directories..."

            summary, names = compute_overall_mota(args.class_to_compute, items)

            motas = [100.0 * sv[13] for sv in summary.values]

            print(f"{'='*10} FINAL OVERALL MOTA SUMMARY {'='*10}")
            print(f"{'-'*35} : MOTA")

            for key, val in zip(names, motas):
                print(f"{str(key):35} : {val:.4f}%")
                if key == "Overall":
                    writer.writerow([str(q), f"{val:.4f}"])
            print("\n")
