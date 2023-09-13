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

import logging
from pathlib import Path
from typing import Dict, List, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from compressai_vision.registry import register_codec

from .common import FeatureTensorCodingType
from .hls import SequenceParameterSet, parse_feature_tensor_coding_type
from .inter import inter_coding, inter_decoding
from .intra import intra_coding, intra_decoding
from .tools import feature_channel_suppression, suppression_optimization

encode_feature_tensor = {
    FeatureTensorCodingType.I_TYPE: intra_coding,
    FeatureTensorCodingType.PB_TYPE: inter_coding,
}

decode_feature_tensor = {
    FeatureTensorCodingType.I_TYPE: intra_decoding,
    FeatureTensorCodingType.PB_TYPE: inter_decoding,
}


def iterate_list_of_tensors(data: Dict):
    list_of_features_sets = list(data.values())
    list_of_keys = list(data.keys())

    num_feature_sets = list_of_features_sets[0].size(0)

    if any(fs.size(0) != num_feature_sets for fs in list_of_features_sets):
        raise ValueError("Feature set items must have the same number of features sets")

    for e, current_feature_set in enumerate(
        tqdm(zip(*list_of_features_sets), total=num_feature_sets)
    ):
        yield e, dict(zip(list_of_keys, current_feature_set))


@register_codec("cfp_codec")
class CFP_CODEC(nn.Module):
    """
    CfP  encoder
    """

    def __init__(
        self,
        **kwargs,
    ):
        self.logger = logging.getLogger(self.__class__.__name__)

        self.enc_cfg = kwargs["encoder_config"]
        self.suppression_cfg = self.enc_cfg["feature_channel_suppression"]

        self.split_layer_list = kwargs["vision_model"].split_layer_list
        self.deeper_features_for_accuracy_proxy = kwargs[
            "vision_model"
        ].deeper_features_for_accuracy_proxy

        self.device = kwargs["vision_model"].device
        self.eval_encode = kwargs["eval_encode"]

        self._sanity_check_for_configuration()

        self.verbosity = kwargs["verbosity"]
        logging_level = logging.WARN
        if self.verbosity == 1:
            logging_level = logging.INFO
        if self.verbosity >= 2:
            logging_level = logging.DEBUG

        self.logger.setLevel(logging_level)

        # encoder parameters & buffers
        self.reset()

    def reset(self):
        self.feature_set_order_count = -1
        self.decoded_tensor_buffer = []
        # self._bitstream_path = None
        self._bitstream_fd = None
        self._is_enc_cfg_printed = False

    @property
    def qp_value(self):
        return self.enc_cfg["qp"]

    @property
    def eval_encode_type(self):
        return self.eval_encode

    def set_bitstream_handle(self, fname, mode="rb"):
        # self._bitstream_path = self.codec_output_dir / f"{fname}"
        fd = self.open_bitstream_file(fname, mode)
        return fd

    def open_bitstream_file(self, path, mode="rb"):
        self._bitstream_fd = open(path, mode)
        return self._bitstream_fd

    def close_files(self):
        if self._bitstream_fd:
            self._bitstream_fd.close()

    def _sanity_check_for_configuration(self):
        suppression_cfgs = self.enc_cfg["feature_channel_suppression"]

        if suppression_cfgs["manual_cluster"] is True:
            if len(suppression_cfgs["n_clusters"]) == 0:
                self.logger.warning(
                    "No clusters provided, manual_cluster is True though.\nFull feature channels will be coded by default"
                )
                self.enc_cfg["feature_channel_suppression"]["n_clusters"] = dict(
                    zip(
                        self.split_layer_list,
                        [None for _ in range(len(self.split_layer_list))],
                    )
                )
        self.downscale = self.suppression_cfg["downscale"]
        self.min_nb_channels_for_group = self.suppression_cfg[
            "min_nb_channels_for_group"
        ]

        self.qp = self.enc_cfg["qp"]
        self.qp_density = self.enc_cfg["qp_density"]
        assert 0 < self.qp_density <= 5, "0 < QP_DENSITY <= 5"

        self.dc_qp_offset = self.enc_cfg["dc_qp_offset"]
        self.dc_qp_density_offset = self.enc_cfg["dc_qp_density_offset"]
        assert (
            self.qp_density + self.dc_qp_density_offset
        ) <= 5, "DC_QP_DENSITY_OFFSET can't be more than (5-qp_density)"

        self.layer_qp_offsets = self.enc_cfg["layer_qp_offsets"]
        if len(self.layer_qp_offsets) >= len(self.split_layer_list):
            self.logger.warning(
                f"Number of qp offsets for layers must be less than total number of feature layers, but got {len(self.layer_qp_offsets)} >= {len(self.split_layer_list)}\n To avoid error, the first {len(self.split_layer_list)-1} offsets will be considered."
            )
            self.layer_qp_offsets = self.layer_qp_offsets[
                : (len(self.split_layer_list) - 1)
            ]

        assert (
            self.enc_cfg["qp"] is not None
        ), "Please provide a QP value!"  # TODO: @eimran maybe run the process to get uncmp result

    def _print_enc_cfg(self, enc_cfg: Dict, lvl: int = 0):
        log_str = ""
        if lvl == 0 and self._is_enc_cfg_printed is True:
            return

        for key, val in enc_cfg.items():
            if isinstance(val, Dict):
                log_str += f"\n {' '*lvl}{'-' * lvl} {key} <"
                log_str += self._print_enc_cfg(val, (lvl + 1))
            else:
                sp = f"<{35-(lvl<<1)}s"
                log_str += f"\n {' '*lvl}{'-' * lvl} {str(key):{sp}} : {val}"

        if lvl == 0:
            intro = f"{'='*10} Encoder Configurations {'='*10}"
            endline = f"{'='*len(intro)}"
            log_str = f"\n {intro}" + log_str + f"\n {endline}" + "\n\n"
            self.logger.info(log_str)

            self._is_enc_cfg_printed = True

        return log_str

    def encode(
        self,
        input: Dict,
        codec_output_dir,
        bitstream_name,
        file_prefix: str = "",
    ) -> Dict:
        hls_header_bytes = 0
        bytes_per_ftensor_set = []

        self.logger.info("Encoding starts...")

        self._print_enc_cfg(self.enc_cfg)

        # check Layers lengths
        layer_nbframes = [
            layer_data.size()[0] for _, layer_data in input["data"].items()
        ]
        assert all(n == layer_nbframes[0] for n in layer_nbframes)
        nbframes = layer_nbframes[0]
        # nbframes = 2  # for debugging

        if file_prefix == "":
            file_prefix = f"{codec_output_dir}/{bitstream_name}"
        else:
            file_prefix = f"{codec_output_dir}/{bitstream_name}-{file_prefix}"
        bitstream_path = f"{file_prefix}.bin"

        bitstream_fd = self.set_bitstream_handle(bitstream_path, "wb")

        # parsing encoder configurations
        intra_period = self.enc_cfg["intra_period"]
        got_size = self.enc_cfg["group_of_tensor"]
        n_bits = 8

        sps = SequenceParameterSet()
        sps.digest(**input)

        # write sps
        hls_header_bytes = sps.write(
            bitstream_fd,
            nbframes,
            downscale_flag=self.downscale,
            qp=self.qp,
            qp_density=self.qp_density,
            layer_qp_offsets=self.layer_qp_offsets,
            dc_qp_offset=self.dc_qp_offset,
            dc_qp_density_offset=self.dc_qp_density_offset,
        )

        bytes_total = hls_header_bytes
        for e, ftensors in iterate_list_of_tensors(input["data"]):
            # counting one for the input
            self.feature_set_order_count += 1  # the same concept as poc

            eFTCType = FeatureTensorCodingType.PB_TYPE
            # All intra when intra_period == -1
            if intra_period == -1 or (self.feature_set_order_count % intra_period) == 0:
                eFTCType = FeatureTensorCodingType.I_TYPE

                ch_clct_by_group, scales_for_layers = suppression_optimization(
                    self.suppression_cfg,
                    input["input_size"],
                    ftensors,
                    self.deeper_features_for_accuracy_proxy,
                    logger=self.logger,
                )

            (
                ftensors_to_code,
                all_channels_coding_modes,
            ) = feature_channel_suppression(
                ftensors,
                ch_clct_by_group,
                scales_for_layers,
                self.min_nb_channels_for_group,
            )

            coded_ftensor_bytes, recon_feature_channels = encode_feature_tensor[
                eFTCType
            ](
                sps,
                ftensors_to_code,
                all_channels_coding_modes,
                scales_for_layers,
                bitstream_fd,
            )

            bytes_total += coded_ftensor_bytes

            bytes_per_ftensor_set.append(bytes_total)

            bytes_total = 0

        self.close_files()

        return {
            "bytes": bytes_per_ftensor_set,
            "bitstream": bitstream_path,
        }

    def decode(
        self,
        input: str,
        codec_output_dir: str = "",
        file_prefix: str = "",
    ):
        del codec_output_dir  # used in other codecs that write log files
        del file_prefix
        self.logger.info("Decoding starts...")

        output = {}

        bitstream_fd = self.open_bitstream_file(input, "rb")

        sps = SequenceParameterSet()

        # read sequence parameter set
        sps.read(bitstream_fd)

        output = {
            "org_input_size": {
                "height": sps.org_input_height,
                "width": sps.org_input_width,
            },
            "input_size": [(sps.input_height, sps.input_width)],
        }

        # temporary tag name
        # it should be replaced outside of decoder with correct name tag to be compatible with NN-Part2
        ftensor_tags = [i for i in range(sps.size_of_feature_set)]

        recon_ftensors = dict(zip(ftensor_tags, [[] for _ in range(len(ftensor_tags))]))
        for ftensor_set_idx in tqdm(range(sps.nbframes)):
            # read coding type
            eFTCType = parse_feature_tensor_coding_type(bitstream_fd)
            res = decode_feature_tensor[eFTCType](sps, bitstream_fd)

            for tlist, item in zip(recon_ftensors.values(), res.values()):
                tlist.append(item)

        self.close_files()

        output["data"] = recon_ftensors

        return output
