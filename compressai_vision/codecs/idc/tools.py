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

import io
import math
from typing import Any, Callable, Dict, List

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as T
from scipy.cluster.hierarchy import cut_tree, linkage
from torch import Tensor

__all__ = [
    "feature_channel_suppression",
    "search_for_N_clusters",
]


def compute_gram_matrix(A: Tensor):
    assert A.dim() == 3
    C, H, W = A.shape

    feature_vectors = A.view(C, H * W).contiguous()
    matrix = torch.mm(feature_vectors, feature_vectors.t())

    return matrix.div(A.numel())


def hierarchical_clustering(A: Tensor, method="ward"):
    array2D = A.detach().cpu().numpy()
    return linkage(array2D, method)


def compute_representation_channel(fchannels: Tensor, mode):
    # TODO [eimran : compute represetnation channel by various mode]
    assert fchannels.dim() == 3
    # it can be different, not necessarily to be 'mean'
    return fchannels.mean(dim=0)


def collect_channels(ftensor: Tensor, channel_groups: List, mode=""):
    unique_groups_ids = np.unique(channel_groups)

    cluster_dict = {}
    represetnations = []
    for e, group_id in enumerate(unique_groups_ids):
        cluster_dict[e] = np.where(channel_groups == group_id)[0].tolist()

        # rep_ch = compute_representation_channel(ftensor[cluster_dict[e]], mode)
        # represetnations.append(rep_ch)

    # cluster_dict = dict(sorted(cluster_dict.items(), key=lambda item: len(item[1])))

    return cluster_dict


def feature_channel_suppression(
    ftensors: Dict, channel_groups: Dict, mode="", n_bits=8
):
    # tensor_min_max = {}
    feature_channels_to_code = {}
    all_channels_coding_groups = {}

    # TODO (frcape) - temporary
    min_nb_channels_per_cluster = 3

    for tag, ftensor in ftensors.items():
        assert tag in channel_groups
        channel_collections = channel_groups[tag]

        rep_ftensor = {}
        channels_coding_groups = np.full(ftensor.shape[0], -1)
        for group_label, channels in channel_collections.items():
            if len(channels) >= min_nb_channels_per_cluster:  # group coded
                channels_coding_groups[channels] = group_label
                rep_ch = compute_representation_channel(ftensor[channels], mode)

                temp_label = min(channels)
                rep_ftensor[temp_label] = rep_ch
            else:  # self coded
                for ch_id in channels:
                    rep_ftensor[ch_id] = ftensor[ch_id]

        rep_ftensor = dict(sorted(rep_ftensor.items(), key=lambda item: item[0]))
        rep_ftensor = list(rep_ftensor.values())

        # est_rep_ftensor, tmin, tmax = forward_min_max_normalization(torch.stack(rep_ftensor), n_bits)

        # feature_channels_to_code[tag] = est_rep_ftensor
        feature_channels_to_code[tag] = torch.stack(rep_ftensor)
        all_channels_coding_groups[tag] = channels_coding_groups
        # tensor_min_max[tag] = (tmin, tmax)

    return feature_channels_to_code, all_channels_coding_groups


def search_for_N_clusters(
    feature_set: Dict,
    proxy_function: Callable,
    n_cluster: Dict,
    measure_thr=-1,
    mode="",
):
    # Find (sub-)optimal N-cluster to categorize input features by channels.
    # Even with N-channels as representative features,
    # it is expected to have minor accuracy degradation for the downtstream task

    # TODO (fracape) add the following as encoder options
    # TODO: Dynamically generate these number using information theory, maybe a RDO? ^_^
    # hard_coded_cluster_number = {
    #     "p2": 128,
    #     "p3": 128,
    #     "p4": 150,
    #     "p5": 180,
    # }  # OpenImage Det & Seg
    # hard_coded_cluster_number ={105: 128, 90: 256, 75: 512} # yolo
    # hard_coded_cluster_number = {"p2": 40, "p3": 256, "p4": 256, "p5": 256}

    hard_coded_cluster_number = n_cluster

    all_channel_collections_by_cluster = {}
    for tag, ftensor in feature_set.items():
        assert ftensor.dim() == 3

        # Original gram matrix at the split layer
        gm = compute_gram_matrix(ftensor)
        hierarchy = hierarchical_clustering(gm)

        num_clusters = hard_coded_cluster_number[tag]

        clustered_channels = cut_tree(hierarchy, num_clusters)

        channel_collections = collect_channels(ftensor, clustered_channels, mode)

        """
        # anyway we won't let the original number of channels be coded.
        max_categories = int(ftensor.size(0) * 3 // 4)

        # one step deeper toward the end layer
        deeper_ftensor = proxy_function(tag, ftensor.unsqueeze(0))
        anchor_gm = compute_gram_matrix(deeper_ftensor.squeeze(0))

        for i in range(16, max_categories, 4)[::-1]:
            i = ftensor.size(0)
            channel_groups = cut_tree(hierarchy, i)

            rep_ftensor, n_clusters = compute_representation_feature_channels(ftensor, channel_groups, mode)

            tmp_ftensor, tmin, tmax = forward_min_max_normalization(torch.stack(rep_ftensor))
            est_ftensor = generate_feature_channels(tmp_ftensor, tmin, tmax, n_clusters, ftensor.shape, ftensor.device)
   
            deeper_test_ftensor = proxy_function(tag, est_ftensor.unsqueeze(0)).squeeze(0)
            test_gm = compute_gram_matrix(deeper_test_ftensor)
        
            gm_loss = F.mse_loss(anchor_gm, test_gm)

            #if gm_loss > measure_thr:
            #if gm_loss > 1.0e-010:
            #    break
        """

        all_channel_collections_by_cluster[tag] = channel_collections

    return all_channel_collections_by_cluster
