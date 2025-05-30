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


from typing import Any, Callable, Dict, Type, TypeVar

import torch.nn as nn
from torch.utils.data import Dataset
from torchvision import transforms

PIPELINES: Dict[str, Callable[..., nn.Module]] = {}
DATACATALOGS: Dict[str, Callable[..., Any]] = {}
DATASETS: Dict[str, Callable[..., Dataset]] = {}
VISIONMODELS: Dict[str, Callable[..., nn.Module]] = {}
EVALUATORS: Dict[str, Callable[..., nn.Module]] = {}
CODECS: Dict[str, Callable[..., nn.Module]] = {}
MULTASK_CODECS: Dict[str, Callable[..., nn.Module]] = {}

TRANSFORMS: Dict[str, Callable[..., Callable]] = {
    k: v for k, v in transforms.__dict__.items() if k[0].isupper()
}

TDatasetCatalog_b = TypeVar("TDatasetCatalog_b", bound=Any)
TDataset_b = TypeVar("TDataset_b", bound=Dataset)
TVisionModel_b = TypeVar("TVisionModel_b", bound=nn.Module)
TEvaluator_b = TypeVar("TEvaluator_b", bound=nn.Module)
TPipeline_b = TypeVar("TPipeline_b", bound=nn.Module)
TCodec_b = TypeVar("TCodec_b", bound=nn.Module)
TMultaskCodec_b = TypeVar("TMultaskCodec_b", bound=nn.Module)


def register_datacatalog(name: str):
    """Decorator for registering a dataset."""

    def decorator(cls: Type[TDatasetCatalog_b]) -> Type[TDatasetCatalog_b]:
        DATACATALOGS[name] = cls
        return cls

    return decorator


def register_dataset(name: str):
    """Decorator for registering a dataset."""

    def decorator(cls: Type[TDataset_b]) -> Type[TDataset_b]:
        DATASETS[name] = cls
        return cls

    return decorator


def register_vision_model(name: str):
    """Decorator for registering a vision model"""

    def decorator(cls: Type[TVisionModel_b]) -> Type[TVisionModel_b]:
        VISIONMODELS[name] = cls
        return cls

    return decorator


def register_evaluator(name: str):
    """Decorator for registering an evaluator"""

    def decorator(cls: Type[TEvaluator_b]) -> Type[TEvaluator_b]:
        EVALUATORS[name] = cls
        return cls

    return decorator


def register_pipeline(name: str):
    """Decorator for registering a pipeline"""

    def decorator(cls: Type[TPipeline_b]) -> Type[TPipeline_b]:
        PIPELINES[name] = cls
        return cls

    return decorator


def register_codec(name: str):
    """Decorator for registering a codec"""

    def decorator(cls: Type[TCodec_b]) -> Type[TCodec_b]:
        CODECS[name] = cls
        return cls

    return decorator


def register_multask_codec(name: str):
    """Decorator for registering a multi-task codec"""

    def decorator(cls: Type[TMultaskCodec_b]) -> Type[TMultaskCodec_b]:
        MULTASK_CODECS[name] = cls
        return cls

    return decorator
