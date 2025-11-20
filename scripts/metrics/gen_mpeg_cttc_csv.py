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

import numpy as np
import pandas as pd
import utils

from compute_overall_map import compute_overall_mAP
from compute_overall_miou import compute_overall_mIoU
from compute_overall_mot import compute_overall_mota
from curve_fitting import (
    convert_to_monotonic_points_SFU,
    convert_to_monotonic_points_TVD,
)

from compressai_vision.datasets import get_seq_info
from compressai_vision.evaluators.evaluators import BaseEvaluator

DATASETS = ["TVD", "SFU", "OIV6", "HIEVE", "PANDASET"]


def read_df_rec(
    path,
    dataset_prefix,
    seq_list,
    nb_operation_points,
    fn_regex=r"summary.csv",
):
    all_summary_csvs = [f for f in iglob(join(path, "**", fn_regex), recursive=True)]
    if nb_operation_points > 0:
        seq_names = [
            file_path.split(path)[1].split("/")[0] for file_path in all_summary_csvs
        ]
        unique_seq_names = list(np.unique(seq_names))
        for sequence in unique_seq_names:
            assert (
                len([f for f in all_summary_csvs if sequence in f])
                == nb_operation_points
            ), f"Did not find {nb_operation_points} results for {sequence}"

    # Only include specified sequences
    matched_summary_csvs = []
    for seq in seq_list:
        matched = [
            f"{dataset_prefix}{seq}" in summary_csv for summary_csv in all_summary_csvs
        ]
        found_at_least_one = False
        for idx, match in enumerate(matched):
            if match:
                matched_summary_csvs.append([seq, all_summary_csvs[idx]])
                found_at_least_one = True
        assert found_at_least_one, f"Found no summary.csv files for {seq}"

    dfs = []
    for seq, f in matched_summary_csvs:
        df = pd.read_csv(f)
        # Overwrite in dataframe to handle inconsistent name RaceHorsesC_832x480_30 found in summary.csv
        df["Dataset"] = seq
        dfs.append(df)
    return pd.concat(dfs, ignore_index=True)


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
    dict_of_classwise_seq,
    metric="AP",
    gt_folder="annotations",
    nb_operation_points: int = 4,
    skip_classwise: bool = False,
    seq_prefix: str = None,
    dataset_prefix: str = None,
):
    seq_list = []
    [seq_list.extend(sequences) for sequences in dict_of_classwise_seq.values()]

    opts_metrics = {"AP": 0, "AP50": 1, "AP75": 2, "APS": 3, "APM": 4, "APL": 5}
    results_df = read_df_rec(result_path, dataset_prefix, seq_list, nb_operation_points)

    # sort
    sorterIndex = dict(zip(seq_list, range(len(seq_list))))
    results_df["ds_rank"] = results_df["Dataset"].map(sorterIndex)
    results_df.sort_values(["ds_rank", "qp"], ascending=[True, True], inplace=True)
    results_df.drop(columns=["ds_rank"], inplace=True)

    output_df = results_df.copy()
    ## drop columns
    output_df.drop(columns=["fps", "num_of_coded_frame"], inplace=True)

    for classwise_name, classwise_seqs in dict_of_classwise_seq.items():
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
                seq_prefix=seq_prefix,
                dataset_prefix=dataset_prefix,
            )

            assert (
                len(items) > 0
            ), "No evaluation information found in provided result directories..."

            if not skip_classwise:
                summary = compute_overall_mAP(
                    dict_of_classwise_seq[classwise_name], items
                )
                maps = summary.values[0][opts_metrics[metric]]
                class_wise_maps.append(maps)

        if not skip_classwise and nb_operation_points > 0:
            class_wise_results_df = generate_classwise_df(
                results_df, {classwise_name: classwise_seqs}
            )
            class_wise_results_df["end_accuracy"] = class_wise_maps

            output_df = df_append(output_df, class_wise_results_df)

    return output_df


def generate_csv_classwise_video_mota(
    result_path,
    dataset_path,
    dict_of_classwise_seq,
    nb_operation_points: int = 4,
    dataset_prefix: str = None,
):
    seq_list = []
    [seq_list.extend(sequences) for sequences in dict_of_classwise_seq.values()]

    results_df = read_df_rec(result_path, dataset_prefix, seq_list, nb_operation_points)
    results_df = results_df.sort_values(by=["Dataset", "qp"], ascending=[True, True])

    # accuracy in % for MPEG template
    results_df["end_accuracy"] = results_df["end_accuracy"].apply(lambda x: x * 100)

    output_df = results_df.copy()
    ## drop columns
    output_df.drop(columns=["fps", "num_of_coded_frame"], inplace=True)

    for classwise_name, classwise_seqs in dict_of_classwise_seq.items():
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

        if nb_operation_points > 0:
            class_wise_results_df = generate_classwise_df(
                results_df, {classwise_name: classwise_seqs}
            )

            class_wise_results_df["end_accuracy"] = class_wise_motas

            output_df = df_append(output_df, class_wise_results_df)

    return output_df


def generate_csv_classwise_video_miou(
    result_path,
    dataset_path,
    dict_of_classwise_seq,
    nb_operation_points: int = 4,
    dataset_prefix: str = None,
):
    seq_list = []
    [seq_list.extend(sequences) for sequences in dict_of_classwise_seq.values()]

    results_df = read_df_rec(result_path, "", seq_list, nb_operation_points)

    # sort
    sorterIndex = dict(zip(seq_list, range(len(seq_list))))
    results_df["ds_rank"] = results_df["Dataset"].map(sorterIndex)
    results_df.sort_values(["ds_rank", "qp"], ascending=[True, True], inplace=True)
    results_df.drop(columns=["ds_rank"], inplace=True)

    output_df = results_df.copy()
    ## drop columns
    output_df.drop(columns=["fps", "num_of_coded_frame"], inplace=True)

    for classwise_name, classwise_seqs in dict_of_classwise_seq.items():
        class_wise_mious = []
        # rate_range = [-1] if nb_operation_points == 1 else range(nb_operation_points)
        for q in range(nb_operation_points):
            items = utils.search_items(
                result_path,
                dataset_path,
                q,
                classwise_seqs,
                BaseEvaluator.get_miou_eval_info_name,
                by_name=True,
                pandaset_flag=True,
            )

            assert (
                len(items) > 0
            ), "Nothing relevant information found from given directories..."

            miou = compute_overall_mIoU(classwise_name, items)
            class_wise_mious.append(miou)

        matched_seq_names = []
        for seq_info in items:
            name, _, _ = get_seq_info(seq_info[utils.SEQ_INFO_KEY])
            matched_seq_names.append(name)

        class_wise_results_df = generate_classwise_df(
            results_df, {classwise_name: classwise_seqs}
        )

        class_wise_results_df["end_accuracy"] = class_wise_mious

        output_df = df_append(output_df, class_wise_results_df)

    return output_df


def generate_csv(result_path, seq_list, nb_operation_points):
    result_df = read_df_rec(result_path, "", seq_list, nb_operation_points)

    # sort
    result_df = result_df.sort_values(by=["Dataset", "qp"], ascending=[True, True])

    # accuracy in % for MPEG template
    result_df["end_accuracy"] = result_df["end_accuracy"].apply(lambda x: x * 100)

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
        choices=["SFU", "OIV6", "TVD", "HIEVE", "PANDASET"],
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
    parser.add_argument(
        "--no-cactus",
        action="store_true",
        default=False,
        help="exclude Cactus sequence for FCM eval",
    )
    parser.add_argument(
        "--add-non-scale",
        action="store_true",
        default=False,
        help="Add non-scale option using ns_Traffic/ns_BQTerrace with original GT",
    )

    args = parser.parse_args()

    assert (
        args.dataset_name.lower() in Path(args.dataset_path).name.lower()
        and args.dataset_name.lower() in Path(args.result_path).name.lower()
    ), "Please check correspondance between input dataset name and result directory"

    norm_result_path = os.path.normpath(args.result_path) + "/"

    if args.dataset_name == "SFU":
        metric = args.metric
        dataset_prefix = "sfu-hw-"
        class_ab = {
            "CLASS-AB": [
                "Traffic_2560x1600_30",
                "Kimono_1920x1080_24",
                "ParkScene_1920x1080_24",
                "Cactus_1920x1080_50",
                "BasketballDrive_1920x1080_50",
                "BQTerrace_1920x1080_60",
            ]
        }
        if args.mode == "VCM":
            class_ab["CLASS-AB"].remove("Kimono_1920x1080_24")
            class_ab["CLASS-AB"].remove("Cactus_1920x1080_50")
        else:
            assert args.mode == "FCM"
            if args.no_cactus is True:
                class_ab["CLASS-AB"].remove("Cactus_1920x1080_50")

        class_c = {
            "CLASS-C": [
                "BasketballDrill_832x480_50",
                "BQMall_832x480_60",
                "PartyScene_832x480_50",
                "RaceHorses_832x480_30",
            ]
        }
        class_d = {
            "CLASS-D": [
                "BasketballPass_416x240_50",
                "BQSquare_416x240_60",
                "BlowingBubbles_416x240_50",
                "RaceHorses_416x240_30",
            ]
        }
        classes = {**class_ab, **class_c, **class_d}
        if args.mode == "VCM" and args.include_optional:
            class_o = {
                "CLASS-O": [
                    "Kimono_1920x1080_24",
                    "Cactus_1920x1080_50",
                ]
            }
            classes.update(class_o)

        if args.mode == "FCM" and args.add_non_scale:
            class_ab_star = {
                "CLASS-AB*": [
                    "ns_Traffic_2560x1600_30",
                    "ns_BQTerrace_1920x1080_60",
                ]
            }
            classes.update(class_ab_star)

        output_df = generate_csv_classwise_video_map(
            norm_result_path,
            args.dataset_path,
            classes,
            metric,
            args.gt_folder,
            args.nb_operation_points,
            args.mode == "VCM",  # skip classwise evaluation
            seq_prefix="ns_",
            dataset_prefix="sfu-hw-",
        )

        if args.mode == "VCM":
            output_df = convert_to_monotonic_points_SFU(
                output_df,
                non_mono_only=False,
                perf_name="end_accuracy",
                rate_name="bitrate (kbps)",
            )
    elif args.dataset_name == "OIV6":
        output_df = generate_csv(
            norm_result_path, ["MPEGOIV6"], args.nb_operation_points
        )
    elif args.dataset_name == "TVD":
        if args.mode == "FCM":
            tvd_all = {"TVD": ["TVD-01", "TVD-02", "TVD-03"]}
            output_df = generate_csv_classwise_video_mota(
                norm_result_path,
                args.dataset_path,
                tvd_all,
                args.nb_operation_points,
                dataset_prefix="mpeg-",
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

            results_df = read_df_rec(
                norm_result_path, tvd_all, args.nb_operation_points
            )
            results_df = results_df.sort_values(
                by=["Dataset", "qp"], ascending=[True, True]
            )

            # accuracy in % for MPEG template
            results_df["end_accuracy"] = results_df["end_accuracy"].apply(
                lambda x: x * 100
            )

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
        hieve = {
            "HIEVE-1080P": ["hieve-13", "hieve-16"],
            "HIEVE-720P": ["hieve-17", "hieve-18", "hieve-2"],
        }
        output_df = generate_csv_classwise_video_mota(
            norm_result_path,
            args.dataset_path,
            hieve,
            args.nb_operation_points,
            dataset_prefix="mpeg-",
        )
    elif args.dataset_name == "PANDASET":
        pandaset = {
            "PANDAM1": [
                "057",
                "058",
                "069",
                "070",
                "072",
                "073",
                "077",
            ],
            "PANDAM2": [
                "003",
                "011",
                "016",
                "017",
                "021",
                "023",
                "027",
                "029",
                "030",
                "033",
                "035",
                "037",
                "039",
                "043",
                "053",
                "056",
                "097",
            ],
            "PANDAM3": [
                "088",
                "089",
                "090",
                "095",
                "109",
                "112",
                "113",
                "115",
                "117",
                "119",
                "122",
                "124",
            ],
        }

        output_df = generate_csv_classwise_video_miou(
            norm_result_path,
            args.dataset_path,
            pandaset,
            args.nb_operation_points,
            dataset_prefix="pandaset-",
        )
    else:
        raise NotImplementedError

    # save
    final_csv_path = os.path.join(norm_result_path, f"final_{args.dataset_name}.csv")
    output_df.to_csv(final_csv_path, sep=",", encoding="utf-8")
    print(output_df)
    print(f"Final CSV Saved at: {final_csv_path}")
