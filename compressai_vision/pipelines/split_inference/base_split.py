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
import os
import shutil
from enum import Enum
from pathlib import Path
from typing import Dict
from uuid import uuid4 as uuid

import torch
import torch.nn as nn

from compressai_vision.evaluators import BaseEvaluator
from compressai_vision.model_wrappers import BaseWrapper

EXT = ".h5"


class Parts(Enum):
    def __str__(self):
        return str(self.value)

    PreInference = "preinference"
    Encoder = "encoder"
    Decoder = "decoder"
    PostInference = "postinference"
    Evaluation = "evaluation"


""" A schematic for the split-inference pipline

.. code-block:: none

             Fold                                                        Fold
           ┌ ─── ┐                                                     ┌ ─── ┐
           |     |                                                     |     |  
           |     │                                                     |     │
     ┌─────|─────▼─────┐                                         ┌─────|─────▼─────┐
     │                 │     ┌───────────┐     ┌───────────┐     │                 │
     │       Pre-      │     │           │     │           │     │      Post-      │
────►│                 ├────►│  Encoder  ├────►│  Decoder  ├────►│                 ├────►
     │    Inference    │     │           │     │           │     │    Inference    │
     │                 │     └───────────┘     └───────────┘     │                 │
     └─────────────────┘                                         └─────────────────┘

    ──────►──────►──────►──────►──────►───Unfold───►──────►──────►──────►──────►──────►
"""


class BaseSplit(nn.Module):
    def __init__(
        self,
        configs: Dict,
        device: str,
    ):
        super().__init__()

        self.logger = logging.getLogger(self.__class__.__name__)
        self.configs = configs
        self.device = device

        self._caching_folder = None
        self._instant_caching = self._use_caching
        if self._use_caching:
            root_folder = self._create_folder(self.configs["output_dir"])

            if self.configs["cache"].dir is None:
                uid = str(uuid())
                self._caching_folder = self._create_folder(
                    os.path.join(root_folder, uid)
                )
            else:
                self._caching_folder = self.configs["cache"].dir
                assert Path(
                    self._caching_folder
                ).is_dir(), f"can't find {self._caching_folder}"
                self._instant_caching = False

    def encode(self):
        """
        Write your own encoding behaviour including the pre-inference + compression part.

        The input is supposed to be image or video, which can be resized within this function
        before using it as input to the front part of the inference model.

        It is ideal to call this function when carrying ``encoding'' out only.
        """
        raise (AssertionError("virtual"))

    def decode(self):
        """
        Write your own decoding behaviour including the uncompression + the post-inference part.

        The input is supposed to be a bistream(s) to decode with the assigned decoder.

        It is ideal to call this function when carrying ``decoding'' out only.
        """
        raise (AssertionError("virtual"))

    def __call__(self):
        """
        Write your own system behaviour from end-to-end.

        Each input will be drawed from the given dataloader, and evaluated through compression and inference
        to measure the performance in terms of rate versus relevant computer vision metrics.

        To measure the computer vision metrics properly, ``evaluator'' must be provided while creating a class instance.

        It is ideal to call this function when evaluating the overall system performance on a given dataset.
        """
        raise (AssertionError("virtual"))

    def _from_input_to_features(self, vision_model: BaseWrapper, x, name: str = None):
        """run the input according to a specific rquirement for input to encoder"""

        if not self._is_caching(Parts.PreInference):
            out = vision_model.input_to_features(x)
            if "image_id" in x[0]:
                out["image_id"] = x[0]["image_id"]
            return out

        # Caching while processing the pre-inference computation
        _folder = self._create_folder(
            os.path.join(self._caching_folder, str(Parts.PreInference))
        )

        _caching_target = os.path.join(_folder, name + EXT)
        if self._check_cache_file(_caching_target):
            cached = torch.load(_caching_target)
        else:
            out = vision_model.input_to_features(x)
            torch.save(out, _caching_target)
            cached = out

        if "image_id" in x[0]:
            cached["image_id"] = x[0]["image_id"]

        return cached

    def _from_features_to_output(
        self, vision_model: BaseWrapper, x: Dict, name: str = None
    ):
        """Postprocess of possibly encoded/decoded data for various tasks inlcuding for human viewing and machine analytics"""
        if not self._is_caching(Parts.PostInference):
            return vision_model.features_to_output(x)

        # Caching while processing the pre-inference computation
        _folder = self._create_folder(
            os.path.join(self._caching_folder, str(Parts.PostInference))
        )

        _caching_target = os.path.join(_folder, name + EXT)
        if self._check_cache_file(_caching_target):
            cached = torch.load(_caching_target)
        else:
            out = vision_model.features_to_output(x)
            torch.save(out, _caching_target)
            cached = out

        return cached

    def _compress_features(self, codec, x, name: str):
        """
        Inputs: tensors of features
        Returns a list of frame bytes and a bitstream path.
        """

        if not self._is_caching(Parts.Encoder):
            return codec.encode(x, name)

        # Caching while processing encoding the input
        _folder = self._create_folder(
            os.path.join(self._caching_folder, str(Parts.Encoder))
        )

        _caching_target = os.path.join(_folder, name + EXT)
        if self._check_cache_file(_caching_target):
            cached = torch.load(_caching_target)
        else:
            out = codec.encode(x, name)
            torch.save(out, _caching_target)
            cached = out

        return cached

    def _decompress_features(self, codec, x, name: str):
        """
        Inputs: a bitstream path
        Returns reconstructed feature tensors
        """

        if not self._is_caching(Parts.Decoder):
            return codec.decode(x, name)

        # Caching while processing encoding the input
        _folder = self._create_folder(
            os.path.join(self._caching_folder, str(Parts.Decoder))
        )

        _caching_target = os.path.join(_folder, name + EXT)
        if self._check_cache_file(_caching_target):
            cached = torch.load(_caching_target)
        else:
            out = codec.decode(x, name)
            torch.save(out, _caching_target)
            cached = out

        return cached

    def _evaluation(self, evaluator: BaseEvaluator) -> Dict:
        save_path = None
        if self._is_caching(Parts.Evaluation):
            # Save output results
            save_path = self._create_folder(
                os.path.join(self._caching_folder, str(Parts.Evaluation))
            )

        out = evaluator.results(save_path)

        return out

    def _is_caching(self, part: Parts):
        """
        Checking if the correponding flag is true before excuting relevant part.

        """
        return self.configs["cache"].enabled & self.configs["cache"][part.value]

    def _create_folder(self, dir):
        path = Path(dir)
        if not path.is_dir():
            self.logger.info(f"creating {dir}")
            path.mkdir(parents=True, exist_ok=True)

        return dir

    def _check_cache_file(self, path):
        return Path(path).exists() & Path(path).is_file()

    @property
    def _use_caching(self):
        return self.configs["cache"].enabled

    @property
    def _keep_caching(self):
        return self.configs["cache"].keep

    def __del__(self):
        if not "cache" in self.configs:
            return
        if self._keep_caching:
            return
        if not self._use_caching:
            return
        if not self._instant_caching:
            return

        self.logger.debug("removing %s", self._caching_folder)
        shutil.rmtree(self._caching_folder)
