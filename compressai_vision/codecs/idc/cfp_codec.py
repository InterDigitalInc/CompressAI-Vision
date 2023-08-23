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

import struct
from pathlib import Path
from typing import Dict, List, Union

import deepCABAC
import numpy as np
import torch
import torch.nn as nn
from scipy.cluster.hierarchy import cut_tree, linkage
from scipy.stats import norm

from compressai_vision.registry import register_codec


# (TODO) place this in encdec_utils file
def write_uints(fd, values, fmt=">{:d}I"):
    fd.write(struct.pack(fmt.format(len(values)), *values))
    return len(values) * 4


def write_uchars(fd, values, fmt=">{:d}B"):
    fd.write(struct.pack(fmt.format(len(values)), *values))
    return len(values) * 1


def read_uints(fd, n, fmt=">{:d}I"):
    sz = struct.calcsize("I")
    return struct.unpack(fmt.format(n), fd.read(n * sz))


def read_uchars(fd, n, fmt=">{:d}B"):
    sz = struct.calcsize("B")
    return struct.unpack(fmt.format(n), fd.read(n * sz))


def write_bytes(fd, values, fmt=">{:d}s"):
    if len(values) == 0:
        return
    fd.write(struct.pack(fmt.format(len(values)), values))
    return len(values) * 1


def read_bytes(fd, n, fmt=">{:d}s"):
    sz = struct.calcsize("s")
    return struct.unpack(fmt.format(n), fd.read(n * sz))[0]


@register_codec("cfp_codec")
class CFP_CODEC(nn.Module):
    """
    CfP  encoder
    """

    def __init__(
        self,
        **kwargs,
    ):
        self.qp = kwargs["encoder_config"]["qp"]
        self.qp_density = kwargs["encoder_config"]["qp_density"]
        self.eval_encode = kwargs["eval_encode"]

        assert (
            self.qp is not None
        ), "Please provide a QP value!"  # TODO: @eimran maybe run the process to get uncmp result

        self.bitstream_dir = Path(kwargs["bitstream_dir"])
        if not self.bitstream_dir.is_dir():
            self.bitstream_dir.mkdir(parents=True, exist_ok=True)

    @property
    def qp_value(self):
        return self.qp

    @property
    def eval_encode_type(self):
        return self.eval_encode

    def encode(
        self,
        input: Dict,
        file_prefix: str = "",
    ) -> Union[List, Dict]:
        byte_cnt = 0

        # check Layers lengths
        layer_nbframes = [
            layer_data.size()[0] for _, layer_data in input["data"].items()
        ]
        assert all(n == layer_nbframes[0] for n in layer_nbframes)
        nbframes = layer_nbframes[0]
        # nbframes = 2  # for debugging

        # TODO (fracape) add the following as encoder options
        # TODO: Dynamically generate these number using information theory, maybe a RDO? ^_^
        cluster_number = {
            "p2": 128,
            "p3": 128,
            "p4": 150,
            "p5": 180,
        }  # OpenImage Det & Seg
        # cluster_number ={105: 128, 90: 256, 75: 512} # yolo
        # cluster_number = {"p2": 40, "p3": 256, "p4": 256, "p5": 256}
        # cluster_number = {"p2": 80, "p3": 180, "p4": 190, "p5": 2}  # SFU-Traffic
        nb_channels_per_cluster = 3
        channel_centerering = False

        bitstream_path = self.bitstream_dir / f"{file_prefix}.bin"

        # encode header (sequence level)
        # TODO: (create separate function)
        with bitstream_path.open("wb") as f:
            # TODO (fracape) is that needed if we transmit layer sizes?
            # write original image size
            byte_cnt += write_uints(f, input["input_size"][0])

            # write nb layers (1 byte)
            byte_cnt += write_uchars(f, (len(input["data"]),))

            # decoder needs number of frames
            byte_cnt += write_uints(f, (nbframes,))

            # encode QP in bitstream (This should be part of picture and layer parameter set )
            byte_cnt += write_uchars(f, (self.qp,))

            # write layer dims (4 bytes each): nb_channels, H, W
            for layer_name, layer_data in input["data"].items():
                _, C, H, W = layer_data.size()
                byte_cnt += write_uints(f, (C, H, W))

            transmitted_data = []

            # loop over frames
            # TODO for now, all intra configuration only, all the frame processed the same way
            # for i in range(layer_nbframes[0]):
            for i in range(nbframes):
                transmitted_data_frame = {}
                for layer_name, layer_data in input["data"].items():
                    transmitted_data_frame[layer_name] = {}
                    _, C, H, W = layer_data.size()
                    frame_layer = layer_data[i : i + 1, :, :, :]

                    layer_data_np = frame_layer.detach().cpu().numpy()

                    gram_matrix = self._get_gram_matrix(frame_layer)
                    cluster_labels = self._get_cluster_labels(
                        gram_matrix, cluster_number[layer_name]
                    )
                    cluster_dict = self._get_cluster_dict(cluster_labels)

                    # keep clusters that have more than n channels
                    cluster_dict = {
                        key: value
                        for key, value in cluster_dict.items()
                        if len(value) >= nb_channels_per_cluster
                    }

                    # sort and rename keys of cluster dictionary
                    cluster_dict = dict(
                        sorted(
                            cluster_dict.items(), key=lambda x: len(x[1]), reverse=True
                        )
                    )
                    cluster_dict = {
                        str(index): value
                        for index, value in enumerate(cluster_dict.values())
                    }

                    # sigma clipping
                    layer_data_np = self._sigma_clipping(layer_name, layer_data_np)

                    # compute representative sample per cluster
                    (
                        cluster_dict,
                        representative_samples_dict,
                    ) = self._get_representative_sample_from_cluster(
                        cluster_dict, layer_data_np
                    )

                    # representative_samples_dict = self._sigma_clipping_clusters(
                    #     layer_name, representative_samples_dict
                    # )

                    # get actual layer_data_np
                    # replace channel data with representative sample if belong to a cluster
                    # for cluster, indices in enumerate(cluster_dict.values()):
                    #     for ch in indices:
                    #         print(layer_data_np[:, ch, :10, :10])
                    #         layer_data_np[:, ch, :, :] = representative_samples_dict[
                    #             f"{cluster}"
                    #         ]

                    # for each channel, get cluster indices or -1 if self coded
                    channels_coding_modes = np.full(C, -1)
                    for cluster, indices in enumerate(cluster_dict.values()):
                        for ch in indices:
                            channels_coding_modes[ch] = cluster

                    if channel_centerering:
                        (
                            representative_samples_dict,
                            mean_dict,
                        ) = self._subtract_mean_from_each_ch(
                            representative_samples_dict
                        )

                        assert len(representative_samples_dict) == len(mean_dict)

                    # dict with the self coded channels
                    self_coded_channels = []
                    for idx, mode in enumerate(channels_coding_modes):
                        if mode < 0:
                            self_coded_channels.append(layer_data_np[:, idx, :, :])
                    self_coded_channels = np.stack(self_coded_channels, axis=0)

                    # encode number of clusters kept
                    # TODO (fracape) could it be retrieved/derived from channels_coding_modes
                    assert (
                        len(cluster_dict) < 256
                    ), "too many clusters, currenlty coding nb clusters on one byte"
                    write_uchars(f, (len(cluster_dict),))

                    # NOTE (fracape) do we need to transmit the number of channels
                    # could  be known by decoder or transmited in SPS
                    # supposed to be known to decode channels_coding_modes

                    # Not compressing the vector  channels_coding_modes for now: can use cabac and
                    # differential coding
                    byte_cnt += write_uchars(f, channels_coding_modes + 1)

                    # encode representative samples of each cluster
                    (
                        bytes_clusters,
                        stream_clusters,
                        quantized_clusters,
                    ) = self._quantize_and_encode(representative_samples_dict)
                    byte_cnt += bytes_clusters
                    byte_cnt += write_uints(f, (bytes_clusters,))
                    f.write(stream_clusters)

                    (
                        bytes_self_coded,
                        stream_self_coded,
                        quantized_self_coded,
                    ) = self._quantize_and_encode(self_coded_channels)
                    byte_cnt += bytes_self_coded
                    byte_cnt += write_uints(f, (bytes_self_coded,))
                    f.write(stream_self_coded)

                    transmitted_data_frame[layer_name]["original_shape"] = [1, C, H, W]
                    transmitted_data_frame[layer_name][
                        "data_clusters"
                    ] = quantized_clusters
                    transmitted_data_frame[layer_name][
                        "data_self_coded"
                    ] = quantized_self_coded
                    transmitted_data_frame[layer_name]["cluster_dict"] = cluster_dict
                    transmitted_data_frame[layer_name][
                        "channels_coding_modes"
                    ] = channels_coding_modes
                    if channel_centerering:
                        transmitted_data_frame[layer_name]["mean_dict"] = mean_dict

                transmitted_data.append(transmitted_data_frame)

                # H, W = input["org_input_size"]["height"], input["org_input_size"]["width"]
                # print("bpp:")
                # print((byte_cnt * 8) / (W * H))

            # TODO (fracape) give a clearer returned objects for bitstream
            # actual tensors in base, bin file in anchors and quantized data for debug here
            #  homogenize with other codecs
            return {
                "bytes": [
                    byte_cnt,
                ],
                "bitstream": transmitted_data,
            }

    def decode(
        self,
        input: Union[List, Dict],
        file_prefix: str = "",
    ):
        bitstream_path = self.bitstream_dir / f"{file_prefix}.bin"
        assert Path(bitstream_path).is_file()

        #
        channel_centerering = False

        with Path(bitstream_path).open("rb") as f:
            # read sequence header
            original_height, original_width = read_uints(f, 2)
            nb_layers = read_uchars(f, 1)[0]
            nb_frames = read_uints(f, 1)[0]
            qp = read_uchars(f, 1)[0]

            # TODO (fracape) is the layer structure
            # and nb_channel per layer supposed to be known by decoder?
            # can be added in bitstrea
            # model id is tricky, since split point and other infos could be necessary
            model_name = "faster_rcnn_X_101_32x8d_FPN_3x"
            # "faster_rcnn_X_101_32x8d_FPN_3x",
            # "mask_rcnn_X_101_32x8d_FPN_3x",
            # "faster_rcnn_R_50_FPN_3x",
            # "mask_rcnn_R_50_FPN_3x",
            # "jde_1088x608",

            # TODO (fracape) implement  jde case
            if "rcnn" in model_name:
                layer_items = ["p2", "p3", "p4", "p5"]
                assert nb_layers == len(layer_items)

            layer_dimensions = {}
            feature_tensor = {}
            for layer_id in layer_items:
                C, H, W = read_uints(f, 3)
                layer_dimensions[layer_id] = (C, H, W)
                feature_tensor[layer_id] = np.zeros(
                    (nb_frames, C, H, W), dtype=np.float32
                )

            for frame_idx in range(nb_frames):
                for layer_id in layer_items:
                    C, H, W = layer_dimensions[layer_id]

                    # decode nb clusters
                    nb_clusters = read_uchars(f, 1)[0]

                    # decode channel coding modes
                    channels_coding_modes = read_uchars(f, C)
                    # minus one to retrieve original ids
                    channels_coding_modes = list(
                        map(lambda x: x - 1, channels_coding_modes)
                    )

                    # number of self coded channels
                    nb_self_coded_channels = channels_coding_modes.count(-1)

                    nb_bytes = read_uints(f, 1)[0]
                    cluster_stream = f.read(nb_bytes)

                    cluster_samples = self._dequantize_and_decode(
                        cluster_stream, (1, nb_clusters, H, W)
                    )

                    nb_bytes = read_uints(f, 1)[0]
                    self_coded_stream = f.read(nb_bytes)
                    self_coded_channels = self._dequantize_and_decode(
                        self_coded_stream, (1, nb_self_coded_channels, H, W)
                    )

                    # TODO (fracape) continue here

                    # TODO (fracape) to be implemented
                    # if channel_centerering:
                    #     mean_dict = {}
                    #     self_coded_channels = self._add_mean_to_each_ch(
                    #         self_coded_channels, mean_dict
                    #     )
                    self_coded_idx = 0
                    for ch_idx in range(C):
                        if channels_coding_modes[ch_idx] == -1:
                            feature_tensor[layer_id][
                                frame_idx, ch_idx, :, :
                            ] = self_coded_channels[self_coded_idx]
                            self_coded_idx += 1
                        else:
                            feature_tensor[layer_id][
                                frame_idx, ch_idx, :, :
                            ] = cluster_samples[channels_coding_modes[ch_idx]]

                # N_hat, C_hat, H_hat, W_hat = feature_tensor[layer_id].shape

            for layer_id in layer_items:
                feature_tensor.update(
                    {layer_id: torch.from_numpy(feature_tensor[layer_id])}
                )

            return feature_tensor

    def _sigma_clipping(self, layer_name, layer_data):
        mu, sigma = norm.fit(layer_data)
        nb_sigma = 1
        if layer_name == "p4" or layer_name == "p5":
            nb_sigma = 3

        min_clip = np.float32(mu - (nb_sigma * sigma))
        max_clip = np.float32(mu + (nb_sigma * sigma))

        assert max_clip >= 0

        # TODO (fracape) check best method, for now, clip at -max, max
        clip_value = max(abs(min_clip), max_clip)

        layer_data = np.clip(layer_data, -clip_value, clip_value)
        return layer_data

    def _subtract_mean_from_each_ch(self, representative_samples_dict):
        mean_dict = {}
        result = {}
        for cluster_no, representative_sample in representative_samples_dict.items():
            mean_ch = np.mean(representative_sample)
            result[cluster_no] = representative_sample - mean_ch
            mean_dict[cluster_no] = mean_ch
        return result, mean_dict

    def _add_mean_to_each_ch(self, representative_samples_dict, mean_dict):
        result = {}
        for cluster_no, representative_sample in representative_samples_dict.items():
            result[cluster_no] = representative_sample + mean_dict[cluster_no]
        return result

    def _quantize_and_encode(self, channels, maxValue=-1):
        # TODO (fracape) check perf vs cabac per channel
        encoder = deepCABAC.Encoder()  # deepCABAC Encoder
        encoder.initCtxModels(10, 0)  # TODO: @eimran Should we change this?

        # print(channels[0, :10, :10])
        quantizedValues = np.zeros(channels.shape, dtype=np.int32)
        encoder.quantFeatures(
            channels, quantizedValues, self.qp_density, self.qp, 0
        )  # TODO: @eimran change scan order and qp method

        # print(quantizedValues[0, :10, :10])
        encoder.encodeFeatures(quantizedValues, 0, maxValue)
        bs = bytearray(encoder.finish().tobytes())
        total_bytes_spent = len(bs)
        # return quantized values for debugging
        return total_bytes_spent, bs, quantizedValues

    def _dequantize_and_decode(self, bs, data_size):
        _, C, H, W = data_size

        assert isinstance(bs, (bytearray, bytes))

        if not isinstance(bs, bytearray):
            bs = bytearray(bs)

        # need to decode max_value
        max_value = -1

        decoder = deepCABAC.Decoder()
        decoder.initCtxModels(10)
        decoder.setStream(bs)
        quantizedValues = np.zeros((C, H, W), dtype=np.int32)
        decoder.decodeFeatures(quantizedValues, 0, max_value)
        # print(quantizedValues[0, :10, :10])
        recon_features = np.zeros(quantizedValues.shape, dtype=np.float32)
        decoder.dequantFeatures(
            recon_features, quantizedValues, self.qp_density, self.qp, 0
        )  # TODO: @eimran send qp with bitstream
        # print(recon_features[0, :10, :10])
        C_hat, H_hat, W_hat = recon_features.shape
        assert C == C_hat
        assert H == H_hat
        assert W == W_hat

        return recon_features

    def _get_gram_matrix(self, input):
        a, b, c, d = input.size()
        # a=batch size(=1)
        # b=number of feature maps
        # (c,d)=dimension of a feature map

        features = input.view(a * b, c * d)  # resize

        G = torch.mm(features, features.t())  # compute the gram product
        # 'normalize' by dividing by the number of element in each feature maps
        return G.div(a * b * c * d)

    def _get_cluster_labels(self, gram_matrix, n_cluster):
        gram_matrix = gram_matrix.detach().cpu().numpy()
        Z = linkage(gram_matrix, "ward")  # TODO: @eimran change linkage mehod
        labels = cut_tree(Z, n_cluster)
        return labels

    def _get_cluster_dict(self, cluster_labels):
        cluster_dict = {}
        n = 0
        for i in range(len(np.unique(cluster_labels))):
            indices = np.where(cluster_labels == i)[0]
            cluster_dict[n] = list(set(indices.tolist()))
            n = n + 1

        return cluster_dict

    def _get_representative_sample_from_cluster(
        self, cluster_dict, param_arr, method="mean"
    ):
        repr_clusters = []
        if method != "mean":
            raise ValueError("representative sample computation not supported")
        for k, v in list(cluster_dict.items()):
            # TODO: @eimran "mean" works better for now! try other methods
            repr_clusters.append(np.mean([param_arr[0][i][:][:] for i in v], axis=0))

        # if 1 <= len(v) <= 5:  # TODO: @eimran change it?!
        #     repr_sample = np.mean([param_arr[0][i][:][:] for i in v], axis=0)
        #     repr_cluster_dict[k] = repr_sample
        # else:
        #     del cluster_dict[k]
        return cluster_dict, np.stack(repr_clusters, axis=0)
