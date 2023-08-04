import math
from typing import Dict, Tuple

import fcvcmCABAC
import numpy as np
import torch
import torch.nn as nn
from scipy.cluster.hierarchy import cut_tree, linkage

from compressai_vision.codec import Bypass
from compressai_vision.registry import register_codec


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
    ) -> Dict:
        del file_prefix

        total_bytes = 0
        result_data_dict = {}

        # TODO: Dynamicly generate these number using information theory, maybe a RDO? ^_^
        cluster_number = {
            "p2": 128,
            "p3": 128,
            "p4": 150,
            "p5": 180,
        }  # OpenImage Det & Seg
        # cluster_number ={105: 128, 90: 256, 75: 512} # yolo
        # cluster_number = {"p2": 40, "p3": 256, "p4": 256, "p5": 256}
        # cluster_number = {"p2": 80, "p3": 180, "p4": 190, "p5": 2}  # SFU-Traffic

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
                representative_samples_dict,
            ) = self._get_representative_sample_from_cluster(
                cluster_dict, layer_data_np
            )

            total_bytes_spent, compressed_dict = self._qunatize_and_encode(
                representative_samples_dict
            )
            total_bytes += total_bytes_spent

            result_data_dict[layer_name] = {}
            result_data_dict[layer_name]["original_shape"] = [N, C, H, W]
            result_data_dict[layer_name]["data"] = compressed_dict
            result_data_dict[layer_name]["side_information"] = cluster_dict

        # H, W = input['org_input_size']['height'], input['org_input_size']['width']
        # print((total_bytes * 8) / (W * H))

        return {
            "bytes": [
                total_bytes,
            ],
            "bitstream": result_data_dict,
        }

    def _qunatize_and_encode(self, representative_samples_dict):
        total_bytes_spent = 0
        compressed = {}
        for cluster_no, derived_fmap in representative_samples_dict.items():
            encoder = fcvcmCABAC.Encoder()  # deepCABAC Encoder
            encoder.initCtxModels(10, 0)  # TODO: @eimran Should we change this?
            quantizedValues = np.zeros(derived_fmap.shape, dtype=np.int32)
            encoder.quantFeatures(
                derived_fmap, quantizedValues, self.qp_density, self.qp, 0
            )  # TODO: @eimran change scan order and qp method
            encoder.encodeFeatures(quantizedValues, 0)
            bs = bytearray(encoder.finish().tobytes())
            compressed[cluster_no] = bs
            total_bytes_spent += len(bs)
        return total_bytes_spent, compressed

    def _dequnatize_and_decode(self, compressed_dict, original_shape):
        N, C, H, W = original_shape
        representative_samples_dict = {}

        for cluster_no, compressed_fmap in compressed_dict.items():
            decoder = fcvcmCABAC.Decoder()
            decoder.initCtxModels(10)
            decoder.setStream(compressed_fmap)
            quantizedValues = np.zeros((H, W), dtype=np.int32)
            decoder.decodeFeatures(quantizedValues, 0)
            recon_features = np.zeros(quantizedValues.shape, dtype=np.float32)
            decoder.dequantFeatures(
                recon_features, quantizedValues, self.qp_density, self.qp, 0
            )  # TODO: @eimran send qp with bitstream
            H_hat, W_hat = recon_features.shape
            assert H == H_hat
            assert W == W_hat
            representative_samples_dict[cluster_no] = recon_features

        return representative_samples_dict

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
            representive_samples_dict = self._dequnatize_and_decode(
                compressed_data["data"], compressed_data["original_shape"]
            )
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

    def _get_representative_sample_from_cluster(
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
