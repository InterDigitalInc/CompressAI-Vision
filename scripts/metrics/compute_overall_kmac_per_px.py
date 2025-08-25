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
    seq_wise_results = []
    for cls_seqs in list_of_classwise_seq:
        seq_name = f"mpeg-oiv6-{cls_seqs}"
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

            if not (os.path.exists(summary_path) or os.path.exists(summary_path)):
                continue

            with open(summary_path, mode="r", newline="", encoding="utf-8") as file:
                reader = csv.DictReader(file)
                data = [row for row in reader][0]
                complexity_dict["qp"] = data["qp"]

            with open(comp_path, mode="r", newline="", encoding="utf-8") as file:
                reader = csv.DictReader(file)
                data = [row for row in reader][0]
                del data["Metric"]
                kmac_per_pixels = {k: float(v) for k, v in data.items()}

            for k, v in kmac_per_pixels.items():
                complexity_dict[k] = v

            seq_wise_results.append(complexity_dict)

    return pd.DataFrame(seq_wise_results)


def generate_csv_classwise_video_gmac(dataset_name, result_path, list_of_classwise_seq):
    seq_base_path = [
        f
        for f in os.listdir(result_path)
        if os.path.isdir(os.path.join(result_path, f))
    ]

    seq_wise_results, cls_wise_results = [], []
    for cls_seqs in list_of_classwise_seq:
        for cls_name, seqs in cls_seqs.items():
            complexity_lst_class_wise = []
            if dataset_name == "PANDASET":
                seqs = [seq.replace("PANDA", "") for seq in seqs]

            seq_path = [
                next(name for name in seq_base_path if s in name)
                for s in seqs
                if any(s in name for name in seq_base_path)
            ]

            for seq in seq_path:
                base_path = f"{result_path}/{seq}"
                qps = [
                    f
                    for f in os.listdir(base_path)
                    if os.path.isdir(os.path.join(base_path, f))
                ]
                qps = sorted(qps)
                for idx, qp in enumerate(qps):
                    complexity_dict = {"Dataset": seq, "pp": idx}
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
            for idx in range(len(qps)):
                (
                    nn_part1_lst,
                    ft_reduction_lst,
                    ft_restoration_lst,
                    nn_part2_lst,
                    nbframe_lst,
                ) = (
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
                        nbframe_lst.append(int(seq_data["nb_frame"]))

                total_frame = sum(nbframe_lst)

                cls_wise_result = {
                    "Dataset": cls_name,
                    "pp": idx,
                    "qp": 0,
                    "nn_part1": sum(
                        [
                            kmac * frames
                            for kmac, frames in zip(nn_part1_lst, nbframe_lst)
                        ]
                    )
                    / total_frame,
                    "feature reduction": sum(
                        [
                            kmac * frames
                            for kmac, frames in zip(ft_reduction_lst, nbframe_lst)
                        ]
                    )
                    / total_frame,
                    "feature restoration": sum(
                        [
                            kmac * frames
                            for kmac, frames in zip(ft_restoration_lst, nbframe_lst)
                        ]
                    )
                    / total_frame,
                    "nn_part2": sum(
                        [
                            kmac * frames
                            for kmac, frames in zip(nn_part2_lst, nbframe_lst)
                        ]
                    )
                    / total_frame,
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
        choices=["SFU", "OIV6", "TVD", "HIEVE", "PANDASET"],
        help="CTTC Evaluation Dataset (default: %(default)s)",
    )

    parser.add_argument(
        "--no-cactus",
        action="store_true",
        default=False,
        help="exclude Cactus sequence for FCM eval",
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

        if args.dataset_name == "SFU" and args.no_cactus:
            if "Cactus_1920x1080_50" in class_ab["CLASS-AB"]:
                class_ab["CLASS-AB"].remove("Cactus_1920x1080_50")

        output_df = generate_csv_classwise_video_gmac(
            args.dataset_name,
            args.result_path,
            [class_ab, class_c, class_d],
        )
    elif args.dataset_name == "OIV6":
        oiv6 = ["detection", "segmentation"]
        output_df = generate_csv_classwise_image_gmac(
            args.dataset_name, args.result_path, oiv6
        )
    elif args.dataset_name == "TVD":
        tvd_all = {"TVD": ["TVD-01", "TVD-02", "TVD-03"]}

        output_df = generate_csv_classwise_video_gmac(
            args.dataset_name, args.result_path, [tvd_all]
        )
    elif args.dataset_name == "HIEVE":
        hieve_1080p = {"HIEVE-1080P": ["13", "16"]}
        hieve_720p = {"HIEVE-720P": ["17", "18", "2"]}

        output_df = generate_csv_classwise_video_gmac(
            args.dataset_name, args.result_path, [hieve_1080p, hieve_720p]
        )

    elif args.dataset_name == "PANDASET":
        PANDAM1 = {
            "PANDAM1": [
                "PANDA057",
                "PANDA058",
                "PANDA069",
                "PANDA070",
                "PANDA072",
                "PANDA073",
                "PANDA077",
            ]
        }
        PANDAM2 = {
            "PANDAM2": [
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
            ]
        }
        PANDAM3 = {
            "PANDAM3": [
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
            ]
        }
        seq_list = [
            "PANDA057",
            "PANDA058",
            "PANDA069",
            "PANDA070",
            "PANDA072",
            "PANDA073",
            "PANDA077",
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
        ]
        output_df = generate_csv_classwise_video_gmac(
            args.dataset_name, args.result_path, [PANDAM1, PANDAM2, PANDAM3]
        )

    else:
        raise NotImplementedError

    # save
    final_csv_path = os.path.join(
        args.result_path, f"final_{args.dataset_name}_kmac.csv"
    )
    output_df.to_csv(final_csv_path, sep=",", encoding="utf-8")
    print(output_df)
    print(f"Final CSV Saved at: {final_csv_path}")
