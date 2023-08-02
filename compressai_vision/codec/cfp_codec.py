import math
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
from scipy.cluster.hierarchy import cut_tree, linkage

from compressai_vision.codec import Bypass
from compressai_vision.model_wrappers import BaseWrapper
from compressai_vision.registry import register_codec


@register_codec("cfp_codec")
class CFP_CODEC(Bypass):
    """Encoder / Decoder class for FCVCM CfP"""

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
        """
        CfP  encoder
        Returns the input and calculates its raw size
        """
        total_elements = 0
        # TODO: Dynamicly generate these number using information theory, maybe a RDO? ^_^
        cluster_number = {
            "p2": 128,
            "p3": 128,
            "p4": 150,
            "p5": 180,
        }  # OpenImage Det & Seg
        # cluster_number = {"p2": 80, "p3": 180, "p4": 190, "p5": 2} # SFU-Traffic
        # cluster_number = {"p2": 40, "p3": 256, "p4": 256, "p5": 256}

        del file_prefix  # used in other codecs that write bitstream files
        result_data_dict = input.copy()

        loss = nn.MSELoss()
        gram_matrix_loss = []
        for layer_name, layer_data in input["data"].items():
            N, C, H, W = layer_data.size()
            layer_data_np = layer_data.detach().cpu().numpy()

            gram_matrix = self._get_gram_matrix(layer_data)
            cluster_labels = self._get_cluster_labels(
                gram_matrix, cluster_number[layer_name]
            )  # change cluster no
            cluster_dict = self._get_cluster_dict(cluster_labels)
            representive_samples_dict = self._get_representive_sample_from_cluster(
                cluster_dict, layer_data_np
            )

            for cluster_no, feature_map_no in cluster_dict.items():
                for i in range(N):
                    for j in feature_map_no:
                        layer_data_np[i][j][:][:] = representive_samples_dict[
                            cluster_no
                        ]

            N_hat, C_hat, H_hat, W_hat = layer_data_np.shape

            assert N == N_hat
            assert C == C_hat
            assert H == H_hat
            assert W == W_hat

            for ft in representive_samples_dict.values():
                total_elements += _number_of_elements(ft.shape)

            result_data_dict["data"].update(
                {layer_name: torch.from_numpy(layer_data_np).to("cuda")}
            )
            gram_matrix_hat = self._get_gram_matrix(
                result_data_dict["data"][layer_name]
            )
            gram_matrix_loss.append(
                loss(gram_matrix, gram_matrix_hat).detach().cpu().numpy()
            )

            # assert

        # write input
        total_bytes = total_elements * 4  # 32-bit floating

        total_gram_matrix_loss = np.sum(gram_matrix_loss)
        print(total_gram_matrix_loss)

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
        # print("Decode")
        # for layer_name, layer_data in input["data"].items():
        #     print(layer_name)
        #     print(layer_data.shape)
        return input

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
        for k, v in cluster_dict.items():
            # TODO: @eimran "mean" works better for now! try other methods
            if method == "max":
                repr_sample = np.max([param_arr[0][i][:][:] for i in v], axis=0)
            if method == "mean":
                repr_sample = np.mean([param_arr[0][i][:][:] for i in v], axis=0)
            repr_cluster_dict[k] = repr_sample
        assert len(cluster_dict) == len(
            repr_cluster_dict
        ), "Number of cluster didn't match with number of representative sample"
        return repr_cluster_dict


def _number_of_elements(data: Tuple):
    return math.prod(data)
