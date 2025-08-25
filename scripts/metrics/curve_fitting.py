# The copyright in this software is being made available under the BSD
# License, included below. This software may be subject to other third party
# and contributor rights, including patent rights, and no such rights are
# granted under this license.

# Copyright (c) 2022, ISO/IEC
# All rights reserved.

# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:

#  * Redistributions of source code must retain the above copyright notice,
#    this list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#  * Neither the name of the ISO/IEC nor the names of its contributors may
#    be used to endorse or promote products derived from this software without
#    specific prior written permission.

# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS
# BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF
# THE POSSIBILITY OF SUCH DAMAGE.

# References:
# 1. H. Wang et al., "Improvements of the BD-Rate Metrics Using Monotonic Curve-Fitting Methods," 2024 Picture Coding Symposium (PCS), Taichung, Taiwan, 2024, pp. 1-5 (available online at  https://doi.org/10.1109/PCS60826.2024.10566370)

import copy
import math
import sys

import numpy as np
import pandas as pd

from cubic_polynomial import fit_cubic, func_cubic_1


def _convert_nan_to_zero(values_possibly_nan):
    for i in range(len(values_possibly_nan)):
        if math.isnan(values_possibly_nan[i]):
            values_possibly_nan[i] = 0.0


def convert_to_monotonic_points_SFU(
    seq_results, non_mono_only=True, perf_name="ap", rate_name="kbps"
):
    """
    Functions to convert non-monotonic points to monotonic points.

    Parameters:
    seq_results: DataFrame --  the test results with non-monotonic points.
    non_mono_only: bool -- apply the curve fitting on non-monotonic points only.

    Return:
    monotonic_seq_results: DataFrame -- the test results with monotonic points.
    """
    num_success = 0
    num_non_success = 0
    monotonic_seq_results = pd.DataFrame()
    for i in range(0, len(seq_results), 6):
        sub_df = seq_results.iloc[i : i + 6, :].reset_index(drop=True)

        # get six x values (ap, mota) and y values
        y_values = sub_df[perf_name].tolist()
        _convert_nan_to_zero(y_values)
        y_values.reverse()

        x_values = sub_df[rate_name].tolist()
        x_values.reverse()
        x_values = np.log10(x_values)

        # check if the points are monotonic increase
        is_increasing = all(x < y for x, y in zip(y_values, y_values[1:]))
        if not non_mono_only or (non_mono_only and not is_increasing):
            # curve fitting
            sub_df = _core_curve_fitting_function(
                x_values, y_values, sub_df, num_success, dataset="SFU"
            )

        monotonic_seq_results = pd.concat(
            [monotonic_seq_results, sub_df], ignore_index=True
        )
    print(f"num_success: {num_success}")
    print(f"num_non_success: {num_non_success}")

    return monotonic_seq_results


def convert_to_monotonic_points_Pandaset(seq_results, non_mono_only=True):
    """
    Functions to convert non-monotonic points to monotonic points.

    Parameters:
    seq_results: DataFrame --  the test results with non-monotonic points.
    non_mono_only: bool -- apply the curve fitting on non-monotonic points only.

    Return:
    monotonic_seq_results: DataFrame -- the test results with monotonic points.
    """
    num_success = 0
    num_non_success = 0
    monotonic_seq_results = pd.DataFrame()
    for i in range(0, len(seq_results), 6):
        sub_df = seq_results.iloc[i : i + 6, :].reset_index(drop=True)

        # get six x values and y values
        y_values = sub_df["mIoU"].tolist()
        _convert_nan_to_zero(y_values)
        y_values.reverse()

        x_values = sub_df["kbps"].tolist()
        x_values.reverse()
        x_values = np.log10(x_values)

        # check if the points are monotonic increase
        is_increasing = all(x < y for x, y in zip(y_values, y_values[1:]))
        if not non_mono_only or (non_mono_only and not is_increasing):
            # curve fitting
            sub_df = _core_curve_fitting_function(
                x_values, y_values, sub_df, num_success, dataset="SFU"
            )

        monotonic_seq_results = pd.concat(
            [monotonic_seq_results, sub_df], ignore_index=True
        )
    print(f"num_success: {num_success}")
    print(f"num_non_success: {num_non_success}")

    return monotonic_seq_results


def convert_to_monotonic_points_TVD(seq_results, non_mono_only=True) -> list:
    """
    Functions to convert non-monotonic points to monotonic points.

    Parameters:
    seq_results: DataFrame --  the test results with non-monotonic points.
    non_mono_only: bool -- apply the curve fitting on non-monotonic points only.
    category: string -- indicate the category

    Return:
    monotonic_seq_results: DataFrame -- the test results with monotonic points.
    """
    seq_results_copy = copy.deepcopy(seq_results)
    monotonic_seq_results = []
    num_success = 0
    num_non_success = 0
    for i in range(0, len(seq_results_copy), 6):
        sub_list = seq_results_copy[i : i + 6]
        # get six x values (ap, mota) and y values
        y_values = [float(x[2]) for x in sub_list]
        _convert_nan_to_zero(y_values)
        y_values.reverse()

        x_values = [x[1] for x in sub_list]
        x_values.reverse()
        x_values = np.log10(x_values)

        # check if the points are monotonic increase
        is_increasing = all(x < y for x, y in zip(y_values, y_values[1:]))
        if not non_mono_only or (non_mono_only and not is_increasing):
            # curve fitting
            sub_list = _core_curve_fitting_function(
                x_values, y_values, sub_list, num_success, dataset="TVD"
            )

        monotonic_seq_results.extend(sub_list)

    print(f"num_success: {num_success}")
    print(f"num_non_success: {num_non_success}")

    return monotonic_seq_results


def _core_curve_fitting_function(x_values, y_values, sub_data, num_success, dataset):
    res = fit_cubic(x_values, y_values)
    num_non_success = 0
    if res.success:
        num_success = num_success + 1
        coef = res.x
        y_values_mono = [func_cubic_1(x, coef) for x in x_values]
        # update ap values
        y_values_mono.reverse()
        if dataset == "TVD":
            for j, d in enumerate(sub_data):
                d[2] = y_values_mono[j]
        else:
            sub_data["ap"] = y_values_mono

    else:
        num_non_success = num_non_success + 1

    return sub_data
