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

from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path

import pandas as pd


def generate_csv_classwise_image_gmac(dataset_name, result_path, list_of_classwise_seq):
    prefix = "mpeg-oiv6"
    seq_wise_results = []
    cls_wise_results = []
    for cls_seqs in list_of_classwise_seq:
        complexity_lst_class_wise = []
        seq_name = f"{prefix}-{cls_seqs}"
        base_path = f"{result_path}/{seq_name}"
        qps = [
            f
            for f in os.listdir(base_path)
            if os.path.isdir(os.path.join(base_path, f))
        ]
        qps = sorted(qps)
        for idx, qp in enumerate(qps):
            complexity_dict = {"Dataset": seq_name, "pp": idx}
            comp_path = f"{base_path}/{qp}/evaluation/summary_complexity.csv"
            summary_path = f"{base_path}/{qp}/evaluation/summary.csv"
            with open(summary_path, mode="r", newline="", encoding="utf-8") as file:
                reader = csv.DictReader(file)
                data = [row for row in reader][0]
                nb_frame = 5000
                complexity_dict["qp"] = data["qp"]

            with open(comp_path, mode="r", newline="", encoding="utf-8") as file:
                reader = csv.DictReader(file)
                data = [row for row in reader][0]
                del data["Metric"]
                avg_kmac = {
                    k: float(v) / (float(nb_frame) * 1000) for k, v in data.items()
                }

            for k, v in avg_kmac.items():
                complexity_dict[k] = v

            complexity_lst_class_wise.append(complexity_dict)
            seq_wise_results.append(complexity_dict)

        # class-wise calculation
        nn_part1_lst, ft_reduction_lst, ft_restoration_lst, nn_part2_lst = (
            [],
            [],
            [],
            [],
        )
        for seq_data in complexity_lst_class_wise:
            nn_part1_lst.append(seq_data["nn_part1"])
            ft_reduction_lst.append(seq_data["feature reduction"])
            ft_restoration_lst.append(seq_data["feature restoration"])
            nn_part2_lst.append(seq_data["nn_part2"])

        cls_wise_result = {
            "Dataset": f"{seq_name}-avg",
            "pp": 0,
            "qp": 0,
            "nn_part1": sum(nn_part1_lst) / len(nn_part1_lst),
            "feature reduction": sum(ft_reduction_lst) / len(ft_reduction_lst),
            "feature restoration": sum(ft_restoration_lst) / len(ft_restoration_lst),
            "nn_part2": sum(nn_part2_lst) / len(nn_part2_lst),
        }

        cls_wise_results.append(cls_wise_result)

    results = pd.DataFrame(seq_wise_results + cls_wise_results)

    return results


def generate_csv_classwise_video_gmac(
    dataset_name, result_path, list_of_classwise_seq, seq_lst
):
    if dataset_name == "SFU":
        prefix = "sfu-hw"
    else:
        prefix = "mpeg"

    seq_wise_results = []
    cls_wise_results = []
    for cls_seqs in list_of_classwise_seq:
        for cls_name, seqs in cls_seqs.items():
            complexity_lst_class_wise = []
            for seq in seqs:
                seq_name = [seq_name for seq_name in seq_lst if seq in seq_name][0]
                base_path = f"{result_path}/{prefix}-{seq_name}_val"
                qps = [
                    f
                    for f in os.listdir(base_path)
                    if os.path.isdir(os.path.join(base_path, f))
                ]
                qps = sorted(qps)
                for idx, qp in enumerate(qps):
                    complexity_dict = {"Dataset": seq_name, "pp": idx}
                    comp_path = f"{base_path}/{qp}/evaluation/summary_complexity.csv"
                    summary_path = f"{base_path}/{qp}/evaluation/summary.csv"
                    with open(
                        summary_path, mode="r", newline="", encoding="utf-8"
                    ) as file:
                        reader = csv.DictReader(file)
                        data = [row for row in reader][0]
                        nb_frame = data["num_of_coded_frame"]
                        complexity_dict["qp"] = data["qp"]
                        complexity_dict["nb_frame"] = nb_frame

                    with open(
                        comp_path, mode="r", newline="", encoding="utf-8"
                    ) as file:
                        reader = csv.DictReader(file)
                        data = [row for row in reader][0]
                        del data["Metric"]
                        kmac_per_pixels = {k: float(v) for k, v in data.items()}

                    for k, v in kmac_per_pixels.items():
                        complexity_dict[k] = v

                    complexity_lst_class_wise.append(complexity_dict)
                    seq_wise_results.append(complexity_dict)

            # class-wise calculation
            for idx in range(4):
                nn_part1_lst, ft_reduction_lst, ft_restoration_lst, nn_part2_lst, nbframe_lst = (
                    [],
                    [],
                    [],
                    [],
                    [],
                )
                for seq_data in complexity_lst_class_wise:
                    if seq_data["pp"] == idx:
                        nn_part1_lst.append(seq_data["nn_part1"])
                        ft_reduction_lst.append(seq_data["feature reduction"])
                        ft_restoration_lst.append(seq_data["feature restoration"])
                        nn_part2_lst.append(seq_data["nn_part2"])
                        nbframe_lst.append(int(seq_data['nb_frame']))
                        
                total_frame = sum(nbframe_lst)

                cls_wise_result = {
                    "Dataset": cls_name,
                    "pp": idx,
                    "qp": 0,
                    "nn_part1": sum([kmac*frames for kmac, frames in zip(nn_part1_lst, nbframe_lst)]) / total_frame,
                    "feature reduction": sum([kmac*frames for kmac, frames in zip(ft_reduction_lst, nbframe_lst)]) / total_frame,
                    "feature restoration": sum([kmac*frames for kmac, frames in zip(ft_restoration_lst, nbframe_lst)]) / total_frame,
                    "nn_part2": sum([kmac*frames for kmac, frames in zip(nn_part2_lst, nbframe_lst)]) / total_frame,
                }

                cls_wise_results.append(cls_wise_result)

    results = pd.DataFrame(seq_wise_results + cls_wise_results)

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-r",
        "--result_path",
        required=True,
        help="For example, '.../logs/runs/[pipeline]/[codec]/[datacatalog]/' ",
    )
    parser.add_argument(
        "-dn",
        "--dataset_name",
        required=True,
        default="SFU",
        choices=["SFU", "OIV6", "TVD", "HIEVE"],
        help="CTTC Evaluation Dataset (default: %(default)s)",
    )

    args = parser.parse_args()

    assert args.dataset_name.lower() in Path(args.result_path).name.lower()

    if args.dataset_name == "SFU":
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
            "RaceHorses_832x480_30",
            "BasketballPass_416x240_50",
            "BQSquare_416x240_60",
            "BlowingBubbles_416x240_50",
            "RaceHorses_416x240_30",
        ]

        output_df = generate_csv_classwise_video_gmac(
            args.dataset_name,
            args.result_path,
            [class_ab, class_c, class_d],
            seq_list,
        )
    elif args.dataset_name == "OIV6":
        oiv6 = ["detection", "segmentation"]
        output_df = generate_csv_classwise_image_gmac(
            args.dataset_name, args.result_path, oiv6
        )
    elif args.dataset_name == "TVD":
        tvd_all = {"TVD": ["TVD-01", "TVD-02", "TVD-03"]}
        seq_list = ["TVD-01", "TVD-02", "TVD-03"]

        output_df = generate_csv_classwise_video_gmac(
            args.dataset_name, args.result_path, [tvd_all], seq_list
        )
    elif args.dataset_name == "HIEVE":
        hieve_1080p = {"HIEVE-1080P": ["13", "16"]}
        hieve_720p = {"HIEVE-720P": ["2", "17", "18"]}
        seq_list = [
            "13_1920x1080_30",
            "16_1920x1080_30",
            "17_1280x720_30",
            "18_1280x720_30",
            "2_1280x720_30",
        ]
        output_df = generate_csv_classwise_video_gmac(
            args.dataset_name, args.result_path, [hieve_1080p, hieve_720p], seq_list
        )
        # sort for FCM template - comply with the template provided in wg04n00459

        sorterIndex = dict(zip(seq_list, range(len(seq_list))))
        output_df["ds_rank"] = output_df["Dataset"].map(sorterIndex)
        output_df.sort_values(["ds_rank", "qp"], ascending=[True, True], inplace=True)
        output_df.drop(columns=["ds_rank"], inplace=True)
    else:
        raise NotImplementedError

    # save
    final_csv_path = os.path.join(
        args.result_path, f"final_{args.dataset_name}_kmac.csv"
    )
    output_df.to_csv(final_csv_path, sep=",", encoding="utf-8")
    print(output_df)
    print(f"Final CSV Saved at: {final_csv_path}")
