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
Compute evaluation metrics and generate csv with computed metrics in CTTC format.
"""
from __future__ import annotations

import argparse
import os
from glob import iglob
from os.path import join
from pathlib import Path

import pandas as pd
from compute_overall_map import compute_overall_mAP
from compute_overall_mot import compute_overall_mota
from curve_fitting import (
    convert_to_monotonic_points_SFU,
    convert_to_monotonic_points_TVD,
)

import utils
from compressai_vision.datasets import get_seq_info
from compressai_vision.evaluators.evaluators import BaseEvaluator

DATASETS = ["TVD", "SFU", "OIV6", "HIEVE"]


def read_df_rec(path, fn_regex=r"summary.csv"):
    return pd.concat(
        (pd.read_csv(f) for f in iglob(join(path, "**", fn_regex), recursive=True)),
        ignore_index=True,
    )


def df_append(df1, df2):
    out = pd.concat([df1, df2], ignore_index=True)
    out.reset_index()
    return out


def generate_classwise_df(result_df, classes: dict):
    classwise = pd.DataFrame(columns=result_df.columns)
    classwise.drop(columns=["fps", "num_of_coded_frame"], inplace=True)

    for tag, item in classes.items():
        output = compute_class_wise_results(result_df, tag, item)
        classwise_df = df_append(classwise, output)

    return classwise_df


def compute_class_wise_results(result_df, name, sequences):
    samples = None
    num_points = prev_num_points = -1
    output = pd.DataFrame(columns=result_df.columns)
    output.drop(columns=["fps", "num_of_coded_frame"], inplace=True)

    for seq in sequences:
        d = result_df.loc[(result_df["Dataset"] == seq)]

        if samples is None:
            samples = d
        else:
            samples = df_append(samples, d)

        if prev_num_points == -1:
            num_points = prev_num_points = d.shape[0]
        else:
            assert prev_num_points == d.shape[0]

    samples["length"] = samples["num_of_coded_frame"] / samples["fps"]

    for i in range(num_points):
        # print(f"Set - {i}")
        points = samples.iloc[range(i, samples.shape[0], num_points)]
        total_length = points["length"].sum()

        # print(points)

        new_row = {
            output.columns[0]: [
                name,
            ],
            output.columns[1]: [
                i,
            ],
        }
        for column in output.columns[2:]:
            # this will be recalculated
            if column == "end_accuracy":
                new_row[column] = -1
                continue

            weighted = points[column] * points["length"]
            new_row[column] = [
                (1 / total_length) * weighted.sum(),
            ]

        output = df_append(output, pd.DataFrame(new_row))

    return output


def generate_csv_classwise_video_map(
    result_path,
    dataset_path,
    list_of_classwise_seq,
    seq_list,
    metric="AP",
    gt_folder="annotations",
    nb_operation_points: int = 4,
    skip_classwise: bool = False,
):
    opts_metrics = {"AP": 0, "AP50": 1, "AP75": 2, "APS": 3, "APM": 4, "APL": 5}
    results_df = read_df_rec(result_path)

    # sort
    sorterIndex = dict(zip(seq_list, range(len(seq_list))))
    results_df["ds_rank"] = results_df["Dataset"].map(sorterIndex)
    results_df.sort_values(["ds_rank", "qp"], ascending=[True, True], inplace=True)
    results_df.drop(columns=["ds_rank"], inplace=True)

    output_df = results_df.copy()
    ## drop columns
    output_df.drop(columns=["fps", "num_of_coded_frame"], inplace=True)

    for seqs_by_class in list_of_classwise_seq:
        classwise_name = list(seqs_by_class.keys())[0]
        classwise_seqs = list(seqs_by_class.values())[0]

        class_wise_maps = []
        for q in range(nb_operation_points):
            items = utils.search_items(
                result_path,
                dataset_path,
                q,
                classwise_seqs,
                BaseEvaluator.get_coco_eval_info_name,
                by_name=True,
                gt_folder=gt_folder,
            )

            assert (
                len(items) > 0
            ), "No evaluation information found in provided result directories..."

            if not skip_classwise:
                summary = compute_overall_mAP(classwise_name, items)
                maps = summary.values[0][opts_metrics[metric]]
                class_wise_maps.append(maps)

        if not skip_classwise:
            matched_seq_names = []
            for seq_info in items:
                name, _, _ = get_seq_info(seq_info[utils.SEQ_INFO_KEY])
                matched_seq_names.append(name)

            class_wise_results_df = generate_classwise_df(
                results_df, {classwise_name: matched_seq_names}
            )
            class_wise_results_df["end_accuracy"] = class_wise_maps

            output_df = df_append(output_df, class_wise_results_df)

    # add empty y_psnr column
    output_df.insert(
        loc=4, column="y_psnr", value=["" for i in range(output_df.shape[0])]
    )

    return output_df


def generate_csv_classwise_video_mota(
    result_path,
    dataset_path,
    list_of_classwise_seq,
    nb_operation_points: int = 4,
):
    results_df = read_df_rec(result_path)
    results_df = results_df.sort_values(by=["Dataset", "qp"], ascending=[True, True])

    # accuracy in % for MPEG template
    results_df["end_accuracy"] = results_df["end_accuracy"].apply(lambda x: x * 100)

    output_df = results_df.copy()
    ## drop columns
    output_df.drop(columns=["fps", "num_of_coded_frame"], inplace=True)

    for seqs_by_class in list_of_classwise_seq:
        classwise_name = list(seqs_by_class.keys())[0]
        classwise_seqs = list(seqs_by_class.values())[0]

        class_wise_motas = []
        for q in range(nb_operation_points):
            items = utils.search_items(
                result_path,
                dataset_path,
                q,
                classwise_seqs,
                BaseEvaluator.get_jde_eval_info_name,
            )

            assert (
                len(items) > 0
            ), "Nothing relevant information found from given directories..."

            summary, _ = compute_overall_mota(classwise_name, items)
            mota = summary.values[-1][13] * 100.0
            class_wise_motas.append(mota)

        matched_seq_names = []
        for seq_info in items:
            name, _, _ = get_seq_info(seq_info[utils.SEQ_INFO_KEY])
            matched_seq_names.append(name)

        class_wise_results_df = generate_classwise_df(
            results_df, {classwise_name: matched_seq_names}
        )

        class_wise_results_df["end_accuracy"] = class_wise_motas

        output_df = df_append(output_df, class_wise_results_df)

    # add empty y_psnr column
    output_df.insert(
        loc=4, column="y_psnr", value=["" for i in range(output_df.shape[0])]
    )

    return output_df


def generate_csv(result_path):
    result_df = read_df_rec(result_path)

    # sort
    result_df = result_df.sort_values(by=["Dataset", "qp"], ascending=[True, True])

    # accuracy in % for MPEG template
    result_df["end_accuracy"] = result_df["end_accuracy"].apply(lambda x: x * 100)

    # add empty y_psnr column
    result_df.insert(
        loc=4, column="y_psnr", value=["" for i in range(result_df.shape[0])]
    )

    return result_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-r",
        "--result_path",
        required=True,
        help="For example, '.../logs/runs/[pipeline]/[codec]/[datacatalog]/' ",
    )
    parser.add_argument(
        "-dp",
        "--dataset_path",
        required=True,
        help="For example, '.../fcm_testdata/[dataset]' ",
    )
    parser.add_argument(
        "-dn",
        "--dataset_name",
        required=True,
        default="SFU",
        choices=["SFU", "OIV6", "TVD", "HIEVE"],
        help="CTTC Evaluation Dataset (default: %(default)s)",
    )
    parser.add_argument(
        "--metric",
        required=False,
        default="AP",
        choices=["AP", "AP50"],
        help="Evaluation Metric (default: %(default)s)",
    )
    parser.add_argument(
        "--nb_operation_points",
        type=int,
        default=4,
        help="number of rate points (qps) per sequence / class",
    )
    parser.add_argument(
        "--gt_folder",
        required=False,
        default="annotations",
        help="folder name for ground truth annotation (default: %(default)s)",
    )
    parser.add_argument(
        "--mode",
        default="FCM",
        choices=["FCM", "VCM"],
        help="CTTC/CTC evaluation mode (default: %(default)s)",
    )
    parser.add_argument(
        "--include_optional",
        action="store_true",
        default=False,
        help="Include optional sequences.",
    )

    args = parser.parse_args()

    assert (
        args.dataset_name.lower() in Path(args.dataset_path).name.lower()
        and args.dataset_name.lower() in Path(args.result_path).name.lower()
    ), "Please check correspondance between input dataset name and result directory"

    if args.dataset_name == "SFU":
        metric = args.metric
        class_ab = {
            "CLASS-AB": [
                "Traffic",
                "Kimono",
                "ParkScene",
                "Cactus",
                "BasketballDrive",
                "BQTerrace",
            ]
        }
        if args.mode == "VCM":
            class_ab["CLASS-AB"].remove("Kimono")
            class_ab["CLASS-AB"].remove("Cactus")
        class_c = {
            "CLASS-C": ["BasketballDrill", "BQMall", "PartyScene", "RaceHorses_832x480"]
        }
        class_d = {
            "CLASS-D": [
                "BasketballPass",
                "BQSquare",
                "BlowingBubbles",
                "RaceHorses_416x240",
            ]
        }
        classes = [class_ab, class_c, class_d]
        if args.mode == "VCM" and args.include_optional:
            class_o = {
                "CLASS-O" : [
                    "Kimono",
                    "Cactus",
                ]
            }
            classes.append(class_o)

        seq_list = [
            "Traffic_2560x1600_30",
            "Kimono_1920x1080_24",
            "ParkScene_1920x1080_24",
            "Cactus_1920x1080_50",
            "BasketballDrive_1920x1080_50",
            "BQTerrace_1920x1080_60",
            "BasketballDrill_832x480_50",
            "BQMall_832x480_60",
            "PartyScene_832x480_50",
            "RaceHorsesC_832x480_30",
            "BasketballPass_416x240_50",
            "BQSquare_416x240_60",
            "BlowingBubbles_416x240_50",
            "RaceHorses_416x240_30",
        ]
        if args.mode == "VCM" and not args.include_optional:
            seq_list.remove("Kimono_1920x1080_24")
            seq_list.remove("Cactus_1920x1080_50")

        output_df = generate_csv_classwise_video_map(
            args.result_path,
            args.dataset_path,
            classes,
            seq_list,
            metric,
            args.gt_folder,
            args.nb_operation_points,
            args.mode == "VCM", # skip classwise evaluation
        )

        if args.mode == "VCM":
            output_df = convert_to_monotonic_points_SFU(
                output_df,
                non_mono_only=False,
                perf_name="end_accuracy",
                rate_name="bitrate (kbps)",
            )
    elif args.dataset_name == "OIV6":
        output_df = generate_csv(args.result_path)
    elif args.dataset_name == "TVD":
        if args.mode == "FCM":
            tvd_all = {"TVD": ["TVD-01", "TVD-02", "TVD-03"]}
            output_df = generate_csv_classwise_video_mota(
                args.result_path,
                args.dataset_path,
                [tvd_all],
                args.nb_operation_points,
            )
        else:
            tvd_all = {
                "TVD": [
                    "TVD-01_1",
                    "TVD-01_2",
                    "TVD-01_3",
                    "TVD-02",
                    "TVD-03_1",
                    "TVD-03_2",
                    "TVD-03_3",
                ]
            }

            results_df = read_df_rec(args.result_path)
            results_df = results_df.sort_values(
                by=["Dataset", "qp"], ascending=[True, True]
            )

            # accuracy in % for MPEG template
            results_df["end_accuracy"] = results_df["end_accuracy"].apply(lambda x: x * 100)

            output_df = results_df.copy()
            ## drop columns
            output_df.drop(columns=["fps", "num_of_coded_frame"], inplace=True)

            output_df = convert_to_monotonic_points_SFU(
                output_df,
                non_mono_only=False,
                perf_name="end_accuracy",
                rate_name="bitrate (kbps)",
            )

    elif args.dataset_name == "HIEVE":
        hieve_1080p = {"HIEVE-1080P": ["13", "16"]}
        hieve_720p = {"HIEVE-720P": ["2", "17", "18"]}
        output_df = generate_csv_classwise_video_mota(
            args.result_path,
            args.dataset_path,
            [hieve_1080p, hieve_720p],
            args.nb_operation_points,
        )
        # sort for FCM template - comply with the template provided in wg04n00459
        seq_list = [
            "13_1920x1080_30",
            "16_1920x1080_30",
            "17_1280x720_30",
            "18_1280x720_30",
            "2_1280x720_30",
            "HIEVE-1080P",
            "HIEVE-720",
        ]
        sorterIndex = dict(zip(seq_list, range(len(seq_list))))
        output_df["ds_rank"] = output_df["Dataset"].map(sorterIndex)
        output_df.sort_values(["ds_rank", "qp"], ascending=[True, True], inplace=True)
        output_df.drop(columns=["ds_rank"], inplace=True)
    else:
        raise NotImplementedError

    # save
    final_csv_path = os.path.join(args.result_path, f"final_{args.dataset_name}.csv")
    output_df.to_csv(final_csv_path, sep=",", encoding="utf-8")
    print(output_df)
    print(f"Final CSV Saved at: {final_csv_path}")
