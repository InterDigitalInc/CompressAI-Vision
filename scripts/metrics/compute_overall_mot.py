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

from typing import Dict

import motmetrics as mm
import torch
import utils

from compressai_vision.evaluators.evaluators import MOT_JDE_Eval


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
    return summary, names
