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

r"""
Compute overall MOT over some sequences outputs



"""
from __future__ import annotations

import argparse
from typing import Any, Dict, List

import motmetrics as mm
import torch

import utils
from compressai_vision.evaluators.evaluators import BaseEvaluator, MOT_JDE_Eval

CLASSES = ["TVD", "HIEVE-1080P", "HIEVE-720P"]

SEQS_BY_CLASS = {
    CLASSES[0]: ["TVD-01", "TVD-02", "TVD-03"],
    CLASSES[1]: ["HIEVE-13", "HIEVE-16"],
    CLASSES[2]: ["HIEVE-2", "HIEVE-17", "HIEVE-18"],
}


def get_accumulator_res_for_tvd(item: Dict):
    _gt_pd = MOT_JDE_Eval._load_gt_in_motchallenge(item[utils.GT_INFO_KEY])
    _pd_pd = MOT_JDE_Eval._format_pd_in_motchallenge(
        torch.load(item[utils.EVAL_INFO_KEY])
    )
    acc, ana = mm.utils.CLEAR_MOT_M(_gt_pd, _pd_pd, item[utils.SEQ_INFO_KEY])

    return acc, ana, item[utils.SEQ_NAME_KEY]


def get_accumulator_res_for_hieve(item: Dict):
    _gt_pd = MOT_JDE_Eval._load_gt_in_motchallenge(
        item[utils.GT_INFO_KEY], min_confidence=1
    )
    _pd_pd = MOT_JDE_Eval._format_pd_in_motchallenge(
        torch.load(item[utils.EVAL_INFO_KEY])
    )
    acc = mm.utils.compare_to_groundtruth(_gt_pd, _pd_pd)

    return acc, None, item[utils.SEQ_NAME_KEY]


def search_items(
    result_path: str, dataset_path: str, rate_point_dir: Any, seq_list: List
):
    _ret_list = []
    for seq_name in seq_list:
        seq_num = utils.get_number(seq_name)

        eval_info_path, dname = utils.get_eval_info_path_by_seq_num(
            seq_num, result_path, rate_point_dir, BaseEvaluator.get_jde_eval_info_name
        )
        seq_info_path, seq_gt_path = utils.get_seq_info_path_by_seq_num(
            seq_num, dataset_path
        )

        d = {
            utils.SEQ_NAME_KEY: dname,
            utils.SEQ_INFO_KEY: seq_info_path,
            utils.EVAL_INFO_KEY: eval_info_path,
            utils.GT_INFO_KEY: seq_gt_path,
        }

        _ret_list.append(d)

    return _ret_list


def compute_overall_mota(class_name, items):
    get_accumulator_res = {
        CLASSES[0]: get_accumulator_res_for_tvd,
        CLASSES[1]: get_accumulator_res_for_hieve,
        CLASSES[2]: get_accumulator_res_for_hieve,
    }

    accs = []
    anas = []
    names = []
    for item in items:
        acc, ana, dname = get_accumulator_res[class_name](item)

        accs.append(acc)
        anas.append(ana)
        names.append(dname)

    mh = mm.metrics.create()
    summary = mh.compute_many(
        accs,
        anas=anas,
        names=names,
        metrics=mm.metrics.motchallenge_metrics,
        generate_overall=True,
    )
    rendered_summary = mm.io.render_summary(
        summary, formatters=mh.formatters, namemap=mm.io.motchallenge_metric_names
    )

    print("\n\n")
    print(rendered_summary)
    print("\n")

    names.append("Overall")
    return summary, names


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-r",
        "--result_path",
        required=True,
        help="For example, '.../logs/runs/[pipeline]/[codec]/[datacatalog]/' ",
    )
    parser.add_argument(
        "-p",
        "--rate_point_dir",
        required=False,
        default=None,
        help="Provide rate point directory name under the `result_path' shared accross different sequences, i.e., `.../[qp00]'",
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

    items = search_items(
        args.result_path,
        args.dataset_path,
        args.rate_point_dir,
        SEQS_BY_CLASS[args.class_to_compute],
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
    print("\n")
