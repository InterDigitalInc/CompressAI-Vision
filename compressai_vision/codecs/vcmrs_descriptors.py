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

import os

# VCMRS version to VCM CTC mapping
# v1.0  -> wg4n0663
# v0.12 -> wg4n0638
# v0.11 -> wg4n0592
# v0.10 -> wg4n0543


# === BEGIN === From VCMRSv1.0
def set_descriptor_files(
    data_dir, scenario, cfg, dataset, video_id, TemporalResamplingAdaptiveMethod=None
):
    main_dir = data_dir  # os.path.dirname(os.path.dirname(data_dir)) # MODIFIED
    # print('set_descriptor_files data_dir', data_dir, 'main_dir', main_dir)

    descriptor_variant_roi = "Unified"
    descriptor_dir_roi = os.path.join(
        main_dir, "Descriptors", descriptor_variant_roi, dataset, "ROI"
    )
    # os.makedirs( descriptor_dir_roi, exist_ok=True)
    cfg["RoIDescriptor"] = os.path.join(descriptor_dir_roi, f"{video_id}.txt")

    if (
        TemporalResamplingAdaptiveMethod is not None
        and TemporalResamplingAdaptiveMethod == "resample_based_detection"
    ):
        descriptor_variant_Temporal = "resample_based_detection"
        descriptor_dir_Temporal = os.path.join(
            main_dir, "Descriptors", descriptor_variant_Temporal, dataset, "Temporal"
        )
        os.makedirs(descriptor_dir_Temporal, exist_ok=True)
        cfg["TemporalDescriptor"] = os.path.join(
            descriptor_dir_Temporal, f"{video_id}.txt"
        )
    elif (
        TemporalResamplingAdaptiveMethod is not None
        and TemporalResamplingAdaptiveMethod == "resample_based_tracking"
    ):
        descriptor_variant_Temporal = "resample_based_tracking"
        descriptor_dir_Temporal = os.path.join(
            main_dir, "Descriptors", descriptor_variant_Temporal, dataset, "Temporal"
        )
        os.makedirs(descriptor_dir_Temporal, exist_ok=True)
        cfg["TemporalDescriptor"] = os.path.join(
            descriptor_dir_Temporal, f"{video_id}.csv"
        )

    descriptor_variant_spatial = "Unified"
    descriptor_dir_spatial = os.path.join(
        main_dir, "Descriptors", descriptor_variant_spatial, dataset, "SpatialResample"
    )
    # os.makedirs( descriptor_dir_spatial, exist_ok=True)
    cfg["SpatialDescriptor"] = os.path.join(descriptor_dir_spatial, f"{video_id}.csv")

    descriptor_variant_colorization = "Unified"
    descriptor_dir_colorization = os.path.join(
        main_dir,
        "Descriptors",
        descriptor_variant_colorization,
        dataset,
        "Colorization",
    )
    # os.makedirs( descriptor_dir_colorization, exist_ok=True)
    cfg["ColorizeDescriptorFile"] = os.path.join(
        descriptor_dir_colorization, f"{video_id}.txt"
    )


# === END ===


# Modified from VCM-RS Scripts/utils.py
def get_descriptor_files_vcm_ctc(
    vcmrs_ver,
    data_dir,
    scenario,
    cfg,
    dataset,
    video_id,
    TemporalResamplingAdaptiveMethod=None,
):
    main_dir = data_dir  # os.path.dirname(os.path.dirname(data_dir))
    if vcmrs_ver == "v1.0":
        cfg = {}
        set_descriptor_files(
            data_dir, scenario, cfg, dataset, video_id, TemporalResamplingAdaptiveMethod
        )
        return cfg
    elif vcmrs_ver == "v0.12":
        descriptor_variant = "Unified"
    elif vcmrs_ver == "v0.11":
        descriptor_variant = "TemporalResampleRatio4"
        if scenario == "AI_e2e":
            descriptor_variant = "TemporalResampleOFF"
        elif scenario == "LD_e2e":
            descriptor_variant = "TemporalResampleExtrapolation"
    else:
        assert False, f"Unsupported VCMRS version {vcmrs_ver}"
    descriptor_dir = os.path.join(main_dir, "Descriptors", descriptor_variant, dataset)
    roi_descriptor = os.path.join(descriptor_dir, "ROI", f"{video_id}.txt")
    spatial_descriptor = os.path.join(
        descriptor_dir, "SpatialResample", f"{video_id}.csv"
    )

    descriptors = {
        "RoIDescriptor": roi_descriptor,
        "SpatialDescriptor": spatial_descriptor,
    }
    if vcmrs_ver >= "v0.12":
        colorize_descriptor = os.path.join(
            descriptor_dir, "Colorization", f"{video_id}.txt"
        )
        descriptors["ColorizeDescriptorFile"] = colorize_descriptor

    return descriptors


# For user-generated/loaded descriptors
def get_descriptor_files_generic(
    vcmrs_ver,
    data_dir,
    scenario,
    cfg,
    dataset,
    video_id,
    TemporalResamplingAdaptiveMethod=None,
):
    descriptor_dir = os.path.join(data_dir, "Descriptors", scenario, dataset)
    roi_descriptor = os.path.join(descriptor_dir, "ROI", f"{video_id}.txt")
    spatial_descriptor = os.path.join(
        descriptor_dir, "SpatialResample", f"{video_id}.csv"
    )

    descriptors = {
        "RoIDescriptor": roi_descriptor,
        "SpatialDescriptor": spatial_descriptor,
    }
    if vcmrs_ver >= "v0.12":
        colorize_descriptor = os.path.join(
            descriptor_dir, "Colorization", f"{video_id}.txt"
        )
        descriptors["ColorizeDescriptorFile"] = colorize_descriptor
    if vcmrs_ver >= "v0.10":
        temporal_descriptor = os.path.join(
            descriptor_dir, "Temporal", f"{video_id}.txt"
        )
        descriptors["TemporalDescriptor"] = temporal_descriptor

    return descriptors


def get_descriptor_files(
    descriptor_mode,
    vcmrs_ver,
    data_dir,
    scenario,
    dataset,
    video_id,
    TemporalResamplingAdaptiveMethod=None,
):
    if descriptor_mode == "vcm_ctc":
        return get_descriptor_files_vcm_ctc(
            vcmrs_ver,
            data_dir,
            scenario,
            {},
            dataset,
            video_id,
            TemporalResamplingAdaptiveMethod,
        )
    else:
        return get_descriptor_files_generic(
            vcmrs_ver,
            data_dir,
            scenario,
            {},
            dataset,
            video_id,
            TemporalResamplingAdaptiveMethod,
        )
