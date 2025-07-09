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


from __future__ import annotations

from typing import Any, Callable, Dict, cast

import torch.nn as nn

from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

import compressai_vision.codecs  # noqa: E731
import compressai_vision.evaluators  # noqa: E731
import compressai_vision.model_wrappers  # noqa: E731
import compressai_vision.pipelines  # noqa: E731

from compressai_vision.datasets import DataCatalog
from compressai_vision.registry import (
    CODECS,
    DATACATALOGS,
    DATASETS,
    EVALUATORS,
    MULTASK_CODECS,
    PIPELINES,
    TRANSFORMS,
    VISIONMODELS,
)

from .env import get_env


def configure_conf(conf: DictConfig):
    conf.env = get_env(conf)

    # cuda environments


def create_vision_model(device: str, conf: DictConfig) -> nn.Module:
    if conf.arch.lower() == "none":
        return None

    return VISIONMODELS[conf.arch](device, **conf[conf.arch]).eval()


def create_data_transform(conf: DictConfig) -> transforms.Compose:
    def register_transform_conf(transform_conf: DictConfig) -> Callable:
        name, kwargs = next(iter(transform_conf.items()))
        name = cast(str, name)
        return TRANSFORMS[name](**kwargs)

    return transforms.Compose(
        [register_transform_conf(transform_conf) for transform_conf in conf]
    )


def create_datacatalog(catalog: str, conf: DictConfig) -> DataCatalog:
    return DATACATALOGS[catalog](**conf)


def create_dataset(_type: str, args: Dict) -> Dataset:
    kwargs = args.copy()
    del kwargs["root"], kwargs["dataset_name"], kwargs["imgs_folder"]

    return DATASETS[_type](
        args["root"], args["dataset_name"], args["imgs_folder"], **kwargs
    )


def create_dataloader(conf: DictConfig, device: str, cfg: Any = None) -> DataLoader:
    args = {
        **OmegaConf.to_container(conf.config, resolve=True),
        **OmegaConf.to_container(conf.settings, resolve=True),
    }

    args["dataset"] = create_datacatalog(conf.datacatalog, conf.config)
    args["transforms"] = create_data_transform(conf.transforms)
    args["cfg"] = cfg

    dataset = create_dataset(conf.type, args)

    return DataLoader(
        dataset,
        batch_size=conf["loader"].batch_size,
        num_workers=conf["loader"].num_workers,
        sampler=dataset.sampler,
        collate_fn=dataset.collate_fn,
        shuffle=conf["loader"].shuffle,
        pin_memory=(device == "cuda"),
    )


def create_evaluator(
    conf: DictConfig, catalog: str, datasetname: str, dataset: Dataset
):
    if conf.type is None:
        return None

    return EVALUATORS[conf.type](catalog, datasetname, dataset, **dict(conf))


def create_pipline(conf: DictConfig, device: DictConfig):
    if conf.type == "":
        pipeline_type = conf.name
    else:
        pipeline_type = conf.type + "-" + conf.name

    return PIPELINES[pipeline_type](dict(conf), dict(device))


def create_codec(conf: DictConfig, vision_model: nn.Module, dataset: DictConfig):
    kwargs = OmegaConf.to_container(conf, resolve=True)

    if "device" not in kwargs:
        raise ValueError("Please specify the argument ++codec.device=cpu or cuda.")

    kwargs["vision_model"] = vision_model
    kwargs["dataset"] = dataset
    kwargs = cast(Dict[str, Any], kwargs)
    del kwargs["type"]
    return CODECS[conf.type](**kwargs)


def create_multi_task_codec(conf: DictConfig, vmodels: list, device: str):
    kwargs = OmegaConf.to_container(conf, resolve=True)
    kwargs = cast(Dict[str, Any], kwargs)
    del kwargs["type"]
    kwargs["vmodels"] = vmodels

    return MULTASK_CODECS[conf.type](device, **kwargs)
