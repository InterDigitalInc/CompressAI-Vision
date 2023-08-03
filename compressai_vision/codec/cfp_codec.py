import math
from typing import Dict, Tuple

import fcvcmCABAC
import numpy as np
import torch
from scipy.cluster.hierarchy import cut_tree, linkage

from compressai_vision.codec import Bypass
from compressai_vision.registry import register_codec


@register_codec("cfp_codec")
class CFP_CODEC(Bypass):
    """
    CfP  encoder
    """

    def __init__(
        self,
        **kwargs,
    ):
        super().__init__(**kwargs)

    def encode(
        self,
        input: Dict,
        file_prefix: str = "",
    ) -> Dict:
        del file_prefix

        # TODO: Dynamicly generate these number using information theory, maybe a RDO? ^_^
        # cluster_number = {
        #     "p2": 128,
        #     "p3": 128,
        #     "p4": 150,
        #     "p5": 180,
        # }  # OpenImage Det & Seg
        cluster_number = {"p2": 80, "p3": 180, "p4": 190, "p5": 2}  # SFU-Traffic
        # cluster_number ={105: 64, 90: 256, 75: 512} # yolo
        # cluster_number = {"p2": 40, "p3": 256, "p4": 256, "p5": 256}

        total_elements = 0
        result_data_dict = {}

        for layer_name, layer_data in input["data"].items():
            N, C, H, W = layer_data.size()
            layer_data_np = layer_data.detach().cpu().numpy()

            gram_matrix = self._get_gram_matrix(layer_data)
            cluster_labels = self._get_cluster_labels(
                gram_matrix, cluster_number[layer_name]
            )
            cluster_dict = self._get_cluster_dict(cluster_labels)
            (
                cluster_dict,
                representive_samples_dict,
            ) = self._get_representive_sample_from_cluster(cluster_dict, layer_data_np)

            for ft in representive_samples_dict.values():
                total_elements += _number_of_elements(ft.shape)

            result_data_dict[layer_name] = {}
            result_data_dict[layer_name]["original_shape"] = [N, C, H, W]
            result_data_dict[layer_name]["data"] = representive_samples_dict
            result_data_dict[layer_name]["side_information"] = cluster_dict

        total_bytes = total_elements * 4  # 32-bit floating

        return {
            "bytes": [
                total_bytes,
            ],
            "bitstream": result_data_dict,
        }

    def decode(
        self,
        input: Dict,
        file_prefix: str = "",
    ):
        del file_prefix
        feature_tensor = {}
        for layer_name, compressed_data in input.items():
            N, C, H, W = compressed_data["original_shape"]
            layer_data_np = np.zeros((N, C, H, W), dtype=np.float32)
            representive_samples_dict = compressed_data["data"]
            cluster_dict = compressed_data["side_information"]

            feature_map_no_set = set()
            total_feature_map_no_set = set(range(1, C))

            for cluster_no, feature_map_no in cluster_dict.items():
                for i in range(N):
                    for j in feature_map_no:
                        layer_data_np[i][j][:][:] = representive_samples_dict[
                            cluster_no
                        ]
                        feature_map_no_set.add(j)

            redundant_fmaps = total_feature_map_no_set - feature_map_no_set

            for i in range(N):
                for j in list(redundant_fmaps):
                    layer_data_np[i][j][:][:] = np.zeros((H, W))

            N_hat, C_hat, H_hat, W_hat = layer_data_np.shape

            assert N == N_hat
            assert C == C_hat
            assert H == H_hat
            assert W == W_hat

            feature_tensor.update({layer_name: torch.from_numpy(layer_data_np)})

        return feature_tensor

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

    def _get_representive_sample_from_cluster(
        self, cluster_dict, param_arr, method="mean"
    ):
        repr_cluster_dict = {}
        for k, v in list(cluster_dict.items()):
            # TODO: @eimran "mean" works better for now! try other methods
            if 1 <= len(v) <= 5:  # TODO: @eimran change it?!
                repr_sample = np.mean([param_arr[0][i][:][:] for i in v], axis=0)
                repr_cluster_dict[k] = repr_sample
            else:
                del cluster_dict[k]
        return cluster_dict, repr_cluster_dict


def _number_of_elements(data: Tuple):
    return math.prod(data)
