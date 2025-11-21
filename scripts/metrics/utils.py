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
Common utils for computing overall MOT/mAP over some sequences outputs



"""

from __future__ import annotations

import os
import re

from pathlib import Path

__all__ = [
    "get_seq_number",
]

SEQ_NAME_KEY = "seq_name"
SEQ_INFO_KEY = "seq_info"
EVAL_INFO_KEY = "eval_info"
GT_INFO_KEY = "gt_info"


def get_seq_number(a):
    num = re.findall(r"\d+", a)
    assert (
        len(num) == 1
    ), f"exepcted only single number in the file name, but many in {a}"
    return num


def check_file_validity(_path):
    assert os.path.exists(_path), f"{_path} does not exist"
    assert os.path.isfile(_path), f"{_path} is not file"

    return True


def get_eval_info_path_by_seq_num(seq_num, _path, qidx: int, name_func: callable):
    eval_folder, dname = get_folder_path_by_seq_num(seq_num, _path)

    if qidx != -1:
        folders = Path(eval_folder).glob("qp*")
        sorted_files = sorted(
            folders, key=lambda x: int(re.search(r"qp(-?\d+)", str(x)).group(1))
        )
        eval_folder = sorted_files[qidx]

    eval_info_path = f"{eval_folder}/evaluation/{name_func(dname)}"

    check_file_validity(eval_info_path)

    return eval_info_path, dname


def get_eval_info_path_by_seq_name(
    seq_name, _path, _qidx: int, name_func: callable, dataset_prefix=None
):
    result = get_folder_path_by_seq_name(seq_name, _path, dataset_prefix)
    if result is None:
        return
    eval_folder, _dname = result

    if _qidx != -1:
        folders = Path(eval_folder).glob("qp*")
        sorted_files = sorted(
            folders, key=lambda x: int(re.search(r"qp(-?\d+)", str(x)).group(1))
        )
        if len(sorted_files) < _qidx + 1:
            return None
        eval_folder = sorted_files[_qidx]

    eval_info_path = f"{eval_folder}/evaluation/{name_func(_dname)}"

    check_file_validity(eval_info_path)

    return eval_info_path, _dname


def get_seq_info_path_by_seq_num(seq_num, _path):
    eval_folder, _ = get_folder_path_by_seq_num(seq_num, _path)

    seq_info_path = f"{eval_folder}/seqinfo.ini"
    check_file_validity(seq_info_path)

    gt_path = f"{eval_folder}/gt/gt.txt"
    check_file_validity(gt_path)

    return seq_info_path, gt_path


def get_seq_info_path_by_seq_name(seq_name, path, gt_folder):
    eval_folder, _dname = get_folder_path_by_seq_name(seq_name, path)

    seq_info_path = f"{eval_folder}/seqinfo.ini"
    check_file_validity(seq_info_path)

    gt_path = f"{eval_folder}/{gt_folder}/{_dname}.json"
    check_file_validity(gt_path)

    return seq_info_path, gt_path


def get_seq_info_path_by_seq_name_pandaset(seq_name, path, gt_folder):
    eval_folder, _dname = get_folder_path_by_seq_name(seq_name, path)

    seq_info_path = f"{eval_folder}/seqinfo.ini"
    check_file_validity(seq_info_path)

    ext = ".npz" if path.endswith("PandaSet") else ".json"
    gt_path = f"{eval_folder}/{gt_folder}/{_dname}{ext}"
    check_file_validity(gt_path)

    return seq_info_path, gt_path


def get_folder_path_by_seq_num(seq_num, _path):
    _folder_list = [f for f in Path(_path).iterdir() if f.is_dir()]
    for _name in _folder_list:
        folder_num = get_seq_number(_name.stem)

        if seq_num == folder_num:
            return _name.resolve(), _name.stem

    return None


def get_folder_path_by_seq_name(seq_name, _path, dataset_prefix=None):
    _folder_list = [f for f in Path(_path).iterdir() if f.is_dir()]

    if dataset_prefix is not None:
        seq_name = f"{dataset_prefix}{seq_name}"

    for _name in _folder_list:
        if seq_name in _name.stem:
            return _name.resolve(), _name.stem

    return None


def search_items(
    result_path: str,
    dataset_path: str,
    rate_point: int,
    seq_list: list,
    eval_func: callable,
    by_name=False,
    pandaset_flag=False,
    gt_folder="annotations",
    seq_prefix=None,
    dataset_prefix=None,
):
    _ret_list = []
    for seq_name in seq_list:
        if by_name is True:
            result = get_eval_info_path_by_seq_name(
                seq_name, result_path, rate_point, eval_func, dataset_prefix
            )
            if result is None:
                continue
            eval_info_path, dname = result
            if pandaset_flag is True:
                seq_info_path, seq_gt_path = get_seq_info_path_by_seq_name_pandaset(
                    seq_name, dataset_path, gt_folder
                )
            else:
                _gt_lookup_name = (
                    seq_name.split(seq_prefix)[-1] if seq_prefix else seq_name
                )
                seq_info_path, seq_gt_path = get_seq_info_path_by_seq_name(
                    _gt_lookup_name, dataset_path, gt_folder
                )
        else:  # by number
            seq_num = get_seq_number(seq_name)

            eval_info_path, dname = get_eval_info_path_by_seq_num(
                seq_num, result_path, rate_point, eval_func
            )
            seq_info_path, seq_gt_path = get_seq_info_path_by_seq_num(
                seq_num, dataset_path
            )

        d = {
            SEQ_NAME_KEY: dname,
            SEQ_INFO_KEY: seq_info_path,
            EVAL_INFO_KEY: eval_info_path,
            GT_INFO_KEY: seq_gt_path,
        }

        _ret_list.append(d)

    return _ret_list
