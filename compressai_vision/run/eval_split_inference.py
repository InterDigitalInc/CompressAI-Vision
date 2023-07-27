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

r"""
Evaluate a system performance of end-to-end pipeline.



"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import hydra
import pandas as pd
from omegaconf import DictConfig
from tabulate import tabulate

thisdir = Path(__file__).parent
config_path = thisdir.joinpath("../../cfgs")


from compressai_vision.config import (
    configure_conf,
    create_codec,
    create_dataloader,
    create_evaluator,
    create_pipline,
    create_vision_model,
)


def setup(conf: DictConfig) -> dict[str, Any]:
    configure_conf(conf)

    vision_model = create_vision_model(conf.misc.device, conf.vision_model)
    dataloader = create_dataloader(conf.dataset, conf.misc.device, vision_model.cfg)
    evaluator = create_evaluator(
        conf.evaluator,
        conf.dataset.datacatalog,
        conf.dataset.config.dataset_name,
        dataloader.dataset,
    )

    codec = create_codec(conf.codec, vision_model, conf.dataset.config.dataset_name)

    pipeline = create_pipline(conf.pipeline, conf.misc.device)

    return pipeline, {
        "vision_model": vision_model,
        "codec": codec,
        "dataloader": dataloader,
        "evaluator": evaluator,
    }


def print_specs(pipeline, **kwargs):
    logger = logging.getLogger(__name__)

    title = (
        lambda a: str(a.__class__).split("<class '")[-1].split("'>")[0].split(".")[-1]
    )

    logger.info(
        f"\
                \n {'='*60}\
                \n Pipeline       : {title(pipeline):<30s}\
                \n Vision Model   : {title(kwargs['vision_model']):<30s}\
                \n  -- Cfg        : {kwargs['vision_model'].model_cfg_path}\
                \n  -- Weight     : {kwargs['vision_model'].pretrained_weight_path}\
                \n Codec          : {title(kwargs['codec']):<30s}\
                \n Dataset        : {kwargs['evaluator'].dataset_name} \
                \n  -- Data       : {kwargs['dataloader'].dataset.images_folder} \
                \n  -- Annotation : {kwargs['dataloader'].dataset.annotation_path} \
                \n  -- SEQ-INFO   : {kwargs['dataloader'].dataset.seqinfo_path} \
                \n Evaluator      : {title(kwargs['evaluator']):<30s}\
                \n  -- DataCatalog: {kwargs['evaluator'].datacatalog_name} \
                \n  -- Output Dir : {kwargs['evaluator'].output_dir} \
                \n  -- Output file: {kwargs['evaluator'].output_file_name} \
                \n\n\
    "
    )


@hydra.main(version_base=None, config_path=str(config_path))
def main(conf: DictConfig):
    pipeline, modules = setup(conf)

    # pretty output
    coded_res, performance = pipeline(**modules)
    coded_res_df = pd.DataFrame(coded_res)

    print("=" * 100)
    print(f"Encoding Information [{pipeline}]")
    coded_res_df["file_name"] = coded_res_df["file_name"].apply(lambda x: Path(x).name)
    print(
        tabulate(coded_res_df, headers="keys", tablefmt="fancy_grid", stralign="center")
    )

    print("Evaluation Performance")
    print(tabulate([performance], tablefmt="psql"))

    coded_res, performance = pipeline(**modules)
    coded_res_df = pd.DataFrame(coded_res)

    print("=" * 100)
    print("Encoding Information")
    coded_res_df["file_name"] = coded_res_df["file_name"].apply(lambda x: Path(x).name)
    print(
        tabulate(coded_res_df, headers="keys", tablefmt="fancy_grid", stralign="center")
    )

    print("Evaluation Performance")
    print(tabulate([performance], tablefmt="psql"))

    # summarize results


if __name__ == "__main__":
    main()
