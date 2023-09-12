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
from numba import jit
from scipy.cluster.hierarchy import cut_tree, linkage
from torch import Tensor

__all__ = [
    "feature_channel_suppression",
    "suppression_optimization",
    "rebuild_ftensor",
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


def get_ftensor_suppressed(
    ch_clct_by_group, ftensor, scale, min_nb_channels_for_group, mode=None
):
    rep_ftensor = {}
    sorted_ch_clct_by_group = {}
    coding_modes = np.full(ftensor.shape[0], -1)

    for group_label, ch_ids in ch_clct_by_group.items():
        if len(ch_ids) >= min_nb_channels_for_group:  # group-coded
            coding_modes[ch_ids] = group_label
            rep_ch = compute_representation_channel(ftensor[ch_ids], mode)
            rep_ftensor[min(ch_ids)] = rep_ch
            sorted_ch_clct_by_group[min(ch_ids)] = ch_ids
        else:  # self-coded
            for ch_id in ch_ids:
                rep_ftensor[ch_id] = ftensor[ch_id]
                sorted_ch_clct_by_group[ch_id] = [ch_id]

    sorted_ch_clct_by_group = dict(
        sorted(sorted_ch_clct_by_group.items(), key=lambda item: item[0])
    )
    rep_ftensor = dict(sorted(rep_ftensor.items(), key=lambda item: item[0]))
    rep_ftensor = list(rep_ftensor.values())
    sftensor = torch.stack(rep_ftensor)

    if scale > 1:
        sftensor = F.interpolate(
            sftensor.unsqueeze(0), scale_factor=1 / scale, mode="bicubic"
        ).squeeze(0)

    return sftensor, coding_modes, sorted_ch_clct_by_group


def feature_channel_suppression(
    ftensors: Dict,
    channel_groups: Dict,
    scales_for_layers: Dict,
    min_nb_channels_for_group,
    mode="",
):
    # tensor_min_max = {}
    ftensors_to_code = {}
    all_channels_coding_modes = {}

    for tag, ftensor in ftensors.items():
        assert tag in channel_groups
        channel_collections = channel_groups[tag]
        scale = scales_for_layers[tag]

        sftensor, coding_modes, _ = get_ftensor_suppressed(
            channel_collections, ftensor, scale, min_nb_channels_for_group, mode
        )

        ftensors_to_code[tag] = sftensor
        all_channels_coding_modes[tag] = coding_modes

    return ftensors_to_code, all_channels_coding_modes


def rebuild_ftensor(channel_groups, rep_ftensor, scale_idx, shape):
    C, H, W = shape

    assert rep_ftensor.shape[0] <= C

    if scale_idx > 1:
        rep_ftensor = F.interpolate(
            rep_ftensor.unsqueeze(0), size=(H, W), mode="bicubic"
        ).squeeze(0)

    assert H == rep_ftensor.shape[1] and W == rep_ftensor.shape[2]

    recon_ftensor = torch.zeros((C, H, W), dtype=torch.float32).to(rep_ftensor.device)
    for channels, ftensor in zip(channel_groups, rep_ftensor):
        recon_ftensor[channels] = ftensor

    return recon_ftensor


@jit(nopython=True)
def compare_proposals(tbboxes, abboxes, amargins, alogits):
    compromised_logits = 0
    for tb in tbboxes:
        for ab, margins, nl in zip(abboxes, amargins, alogits):
            gap = np.abs(tb - ab)
            if (gap < margins).all():
                compromised_logits += nl
                break

    return compromised_logits


def update_margins(a_proposals, xy_margin):
    a_wmargins = (a_proposals[:, 2] - a_proposals[:, 0]) * xy_margin
    a_hmargins = (a_proposals[:, 3] - a_proposals[:, 1]) * xy_margin
    return np.stack([a_wmargins, a_hmargins, a_wmargins, a_hmargins], axis=-1)


def find_N_groups_of_channels(
    tag,
    search_base,
    search_distance,
    search_dscale_idx,
    proxy_input,
    proxy_function,
    ftensor,
    anchor,
    hierarchy,
    cvg_thres,
    min_nb_channels_for_group,
    mode,
):
    max_channels = ftensor.shape[0]

    find_optimum = False

    bidx = search_base
    eidx = min(search_base + search_distance, ftensor.shape[0])

    sidx = search_base + (search_distance // 2)
    pidx = search_base

    neat_est_ftensor = None
    neat_proposal_coverage = 1.0
    neat_channel_collections = None
    neat_num_clusters = 0
    neat_num_chs_to_code = 0

    while find_optimum is False:
        channel_groups = cut_tree(hierarchy, sidx)

        channel_collections = collect_channels(ftensor, channel_groups, mode)

        sftensor, _, sorted_ch_clct_by_group = get_ftensor_suppressed(
            channel_collections,
            ftensor,
            (search_dscale_idx + 1),
            min_nb_channels_for_group,
            mode,
        )

        est_ftensor = rebuild_ftensor(
            sorted_ch_clct_by_group.values(),
            sftensor,
            (search_dscale_idx + 1),
            ftensor.shape,
        )

        proxy_input["data"][tag] = est_ftensor
        eval_res = proxy_function(proxy_input)

        e_proposals = np.array(eval_res.proposal_boxes.tensor.detach().cpu())
        counted_logits = compare_proposals(
            e_proposals, anchor["proposals"], anchor["margins"], anchor["norm_logits"]
        )

        current_proposal_coverage = (counted_logits) / anchor["total_logits"]

        if abs(sidx - pidx) <= 2:
            find_optimum = True

            if neat_est_ftensor == None:
                neat_proposal_coverage = current_proposal_coverage
                neat_channel_collections = channel_collections
                neat_est_ftensor = est_ftensor.unsqueeze(0)
                neat_num_clusters = sidx
                neat_num_chs_to_code = sftensor.shape[0]

            return (
                neat_est_ftensor,
                neat_channel_collections,
                neat_num_clusters,
                neat_num_chs_to_code,
                neat_proposal_coverage,
            )

        pidx = sidx
        if current_proposal_coverage >= cvg_thres:
            sidx = max(sidx - ((sidx - bidx) // 2), 0)

            if current_proposal_coverage < neat_proposal_coverage:
                neat_proposal_coverage = current_proposal_coverage
                neat_channel_collections = channel_collections
                neat_est_ftensor = est_ftensor.unsqueeze(0)
                neat_num_clusters = pidx
                neat_num_chs_to_code = sftensor.shape[0]
        else:
            bidx = sidx
            sidx = min(sidx + ((eidx - bidx) // 2), max_channels)


def _manual_clustering(n_clusters: Dict, downscale, ftensors: Dict, mode):
    all_ch_clct_by_group = {}
    all_scales_for_layers = {}
    scale = 2 if downscale else 1
    for tag, ftensor in ftensors.items():
        assert ftensor.dim() == 3

        # Original gram matrix at the split layer
        gm = compute_gram_matrix(ftensor)
        hierarchy = hierarchical_clustering(gm)

        num_clusters = ftensor.shape[0] if n_clusters[tag] is None else n_clusters[tag]
        num_clusters = min(num_clusters, ftensor.shape[0])

        clustered_channels = cut_tree(hierarchy, num_clusters)
        channel_collections = collect_channels(ftensor, clustered_channels, mode)

        all_ch_clct_by_group[tag] = channel_collections
        all_scales_for_layers[tag] = scale

    return all_ch_clct_by_group, all_scales_for_layers


def suppression_optimization(
    enc_cfg: Dict,
    input_img_size,
    ftensors: Dict,
    proxy_function: Callable,
    mode="",
    logger=None,
):
    # Find (sub-)optimal N-cluster to categorize input features by channels.
    # Even with N-channels as representative features,
    # it is expected to have minor accuracy degradation for the downtstream task

    if enc_cfg["manual_cluster"] is True:
        assert "n_clusters" in enc_cfg
        return _manual_clustering(
            enc_cfg["n_clusters"], enc_cfg["downscale"], ftensors, mode
        )

    min_nb_channels_for_group = enc_cfg["min_nb_channels_for_group"]
    xy_margin = enc_cfg["xy_margin"]

    scale_list = (
        [
            0,
        ]
        if enc_cfg["downscale"] is False
        else [0, 1]  # Complexity increases as the list extends
    )
    org_coverage_thres = enc_cfg["coverage_thres"]
    # empirical initial decay
    weight_decay = enc_cfg["coverage_decay"]
    margin_ext = 1 + enc_cfg["xy_margin_decay"]

    proxy_input = {"data": ftensors.copy(), "input_size": input_img_size}
    res = proxy_function(proxy_input)

    # x1, y1, x2, y2 format
    a_proposals = np.array(res.proposal_boxes.tensor.detach().cpu())
    a_logits = np.array(res.objectness_logits.detach().cpu())
    norm_a_logits = np.abs(a_logits) / np.linalg.norm(a_logits)
    total_logits = norm_a_logits.sum()

    a_margins = update_margins(a_proposals, xy_margin)

    anchor = {
        "proposals": a_proposals,
        "margins": a_margins,
        "norm_logits": norm_a_logits,
        "total_logits": total_logits,
    }

    all_ch_clct_by_group = {}
    all_scales_for_layers = {}

    cvg_thres = org_coverage_thres
    for tag, ftensor in ftensors.items():
        assert ftensor.dim() == 3

        # Original gram matrix at the split layer
        gm = compute_gram_matrix(ftensor)
        hierarchy = hierarchical_clustering(gm)

        search_base = 0
        search_distance = ftensor.size(0)

        best_est_ftensor = None
        best_channel_collections = None
        best_num_clusters = 0
        best_num_chs_to_code = 0
        best_scale_idx = 0

        dscale_0_best_cvg = 0

        for dscale_idx in scale_list:
            (
                opt_est_ftensor,
                opt_channel_collections,
                opt_num_cluters,
                opt_num_chs_to_code,
                opt_coverage,
            ) = find_N_groups_of_channels(
                tag,
                search_base,
                search_distance,
                dscale_idx,
                proxy_input,
                proxy_function,
                ftensor,
                anchor,
                hierarchy,
                cvg_thres,
                min_nb_channels_for_group,
                mode,
            )

            if best_est_ftensor is None or (
                (opt_coverage >= (max(dscale_0_best_cvg, org_coverage_thres) * 0.85))
                and dscale_idx > 0
            ):
                best_est_ftensor = opt_est_ftensor
                best_channel_collections = opt_channel_collections
                best_num_clusters = opt_num_cluters
                best_num_chs_to_code = opt_num_chs_to_code
                best_coverage = opt_coverage
                best_scale_idx = dscale_idx

                # update search base
                search_base = max((opt_num_cluters - 16), 0)
                search_distance = min((ftensor.shape[0] - search_base), 32)

                if dscale_idx == 0:
                    dscale_0_best_cvg = best_coverage

        # decay cvg_thres
        cvg_thres = cvg_thres * weight_decay
        xy_margin = xy_margin * margin_ext
        anchor["margins"] = update_margins(a_proposals, xy_margin)

        if logger:
            logger.debug(
                f"{tag} - {best_num_clusters} --> nb chs: {best_num_chs_to_code} dscale: 1/{best_scale_idx+1}- coverage: {best_coverage * 100:.2f} %"
            )

        proxy_input["data"][tag] = best_est_ftensor
        all_ch_clct_by_group[tag] = best_channel_collections
        all_scales_for_layers[tag] = best_scale_idx + 1

    return all_ch_clct_by_group, all_scales_for_layers
