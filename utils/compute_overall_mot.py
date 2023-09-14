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
import copy
import os
import re
import time
from glob import glob
from typing import Any, Dict, List

import motmetrics as mm
import torch

from compressai_vision.evaluators.evaluators import BaseEvaluator, MOT_JDE_Eval

SEQ_NAME_KEY = "seq_name"
SEQ_INFO_KEY = "seq_info"
EVAL_INFO_KEY = "eval_info"
GT_INFO_KEY = "gt_info"

CLASSES = ["TVD", "HIEVE-1080P", "HIEVE-720P"]

SEQS_BY_CLASS = {
    CLASSES[0]: ["TVD-01", "TVD-02", "TVD-03"],
    CLASSES[1]: ["HIEVE-13", "HIEVE-16"],
    CLASSES[2]: ["HIEVE-2", "HIEVE-17", "HIEVE-18"],
}


def get_accumulator_res_for_tvd(item: Dict):
    _gt_pd = MOT_JDE_Eval._load_gt_in_motchallenge(item[GT_INFO_KEY])
    _pd_pd = MOT_JDE_Eval._format_pd_in_motchallenge(torch.load(item[EVAL_INFO_KEY]))
    acc, ana = mm.utils.CLEAR_MOT_M(_gt_pd, _pd_pd, item[SEQ_INFO_KEY])

    return acc, ana, item[SEQ_NAME_KEY]


def get_accumulator_res_for_hieve(item: Dict):
    _gt_pd = MOT_JDE_Eval._load_gt_in_motchallenge(item[GT_INFO_KEY], min_confidence=1)
    _pd_pd = MOT_JDE_Eval._format_pd_in_motchallenge(torch.load(item[EVAL_INFO_KEY]))
    acc = mm.utils.compare_to_groundtruth(_gt_pd, _pd_pd)

    return acc, None, item[SEQ_NAME_KEY]


def get_number(a):
    num = re.findall(r"\d+", a)
    assert (
        len(num) == 1
    ), f"exepcted only single number in the file name, but many in {a}"
    return num


def check_file_validity(_path):
    assert os.path.exists(_path), f"{_path} does not exist"
    assert os.path.isfile(_path), f"{_path} is not file"

    return True


def get_eval_info_path(seq_num, _path, _subdir):
    eval_folder, _dname = get_folder_path(seq_num, _path)

    if _subdir is not None:
        eval_folder = f"{eval_folder}/{_subdir}"

    eval_info_path = (
        f"{eval_folder}/evaluation/{BaseEvaluator.get_eval_info_name(_dname)}"
    )

    check_file_validity(eval_info_path)

    return eval_info_path, _dname


def get_seq_info_path(seq_num, _path):
    eval_folder, _ = get_folder_path(seq_num, _path)

    seq_info_path = f"{eval_folder}/seqinfo.ini"
    check_file_validity(seq_info_path)

    gt_path = f"{eval_folder}/gt/gt.txt"
    check_file_validity(gt_path)

    return seq_info_path, gt_path


def get_folder_path(seq_num, _path):
    _folder_list = os.listdir(_path)

    for _name in _folder_list:
        folder_num = get_number(_name)

        if seq_num == folder_num:
            return f"{_path}/{_name}", _name

    return None


def search_items(
    result_path: str, dataset_path: str, subdirectory: Any, seq_list: List
):
    _ret_list = []
    for seq_name in seq_list:
        seq_num = get_number(seq_name)

        eval_info_path, dname = get_eval_info_path(seq_num, result_path, subdirectory)
        seq_info_path, seq_gt_path = get_seq_info_path(seq_num, dataset_path)

        d = {
            SEQ_NAME_KEY: dname,
            SEQ_INFO_KEY: seq_info_path,
            EVAL_INFO_KEY: eval_info_path,
            GT_INFO_KEY: seq_gt_path,
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
        "-s",
        "--sub_directory",
        required=False,
        default=None,
        help="Provide subdirectory name under the `result_path' shared accross different sequences, i.e., `.../[qp00]'",
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
        args.sub_directory,
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
