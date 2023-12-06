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


import configparser
import copy

import cv2
import numpy as np
import torch
from jde.utils.datasets import letterbox

__all__ = ["JDECustomMapper"]


class JDECustomMapper:
    """
    A callable which takes a dataset dict in CompressAI-Vision generic dataset format, but for JDE Tracking,
    and map it into a format used by the model.

    This is the default callable to be used to map your dataset dict into inference data.

    This callable function refers to
        LoadImages function in Towards-Realtime-MOT/utils/datasets.py at
        <https://github.com/Zhongdao/Towards-Realtime-MOT/blob/master/utils/datasets.py>

        Full license statement can be found at
        <https://github.com/Zhongdao/Towards-Realtime-MOT/blob/master/LICENSE>

    """

    def __init__(self, img_size=[608, 1088]):
        """
        Args:
            img_size: expected input size (Height, Width)
        """

        self.height, self.width = img_size

    def __call__(self, dataset_dict):
        """
        Args:
            dataset_dict (dict): Metadata of one image.

        Returns:
            dict: a format that compressai-vision pipelines accept
        """

        dataset_dict = copy.deepcopy(dataset_dict)
        # the copied dictionary will be modified by code below

        dataset_dict.pop("annotations", None)

        # Read image
        org_img = cv2.imread(dataset_dict["file_name"])  # return img in BGR by default
        dataset_dict["height"], dataset_dict["width"], _ = org_img.shape

        # Padded resize
        image, _, _, _ = letterbox(org_img, height=self.height, width=self.width)

        # convert to RGB & swap axis
        image = image[:, :, ::-1].transpose(2, 0, 1)
        # normalize contiguous array of image
        image = np.ascontiguousarray(image, dtype=np.float32) / 255.0
        # to tensor
        dataset_dict["image"] = torch.as_tensor(image)

        return dataset_dict


def get_seq_info(seq_info_path):
    config = configparser.ConfigParser()
    config.read(seq_info_path)
    fps = config["Sequence"]["frameRate"]
    total_frame = config["Sequence"]["seqLength"]
    name = f'{config["Sequence"]["name"]}_{config["Sequence"]["imWidth"]}x{config["Sequence"]["imHeight"]}_{fps}'
    return name, int(fps), int(total_frame)
