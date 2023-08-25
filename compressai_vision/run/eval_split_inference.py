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

import configparser
import logging
import os
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


title = lambda a: str(a.__class__).split("<class '")[-1].split("'>")[0].split(".")[-1]


def print_specs(pipeline, **kwargs):
    logger = logging.getLogger(__name__)

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

    print_specs(pipeline, **modules)
    eval_encode_type, coded_res, performance = pipeline(**modules)

    # pretty output
    coded_res_df = pd.DataFrame(coded_res)

    print("=" * 100)
    print(f"Encoding Information [Top 5 Rows...][{pipeline}]")
    coded_res_df["file_name"] = coded_res_df["file_name"].apply(lambda x: Path(x).name)
    coded_res_df["total_pixels"] = coded_res_df["org_input_size"].apply(
        lambda x: int(x.split("x")[0]) * int(x.split("x")[1])
    )
    print(
        tabulate(
            coded_res_df.head(5),
            headers="keys",
            tablefmt="fancy_grid",
            stralign="center",
        )
    )

    print("Evaluation Performance")
    print(tabulate([performance], tablefmt="psql"))

    coded_res, performance = pipeline(**modules)

    # pretty output
    coded_res_df = pd.DataFrame(coded_res)

    print("=" * 100)
    print(f"Encoding Information [{pipeline}]")
    coded_res_df["file_name"] = coded_res_df["file_name"].apply(lambda x: Path(x).name)
    coded_res_df["total_pixels"] = coded_res_df["org_input_size"].apply(
        lambda x: int(x.split("x")[0]) * int(x.split("x")[1])
    )
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
    coded_res_df["total_pixels"] = coded_res_df["org_input_size"].apply(
        lambda x: int(x.split("x")[0]) * int(x.split("x")[1])
    )
    print(
        tabulate(
            coded_res_df.head(5),
            headers="keys",
            tablefmt="fancy_grid",
            stralign="center",
        )
    )

    # summarize results
    evaluator_name = _get_evaluator_name(**modules)
    evaluator_filepath = _get_evaluator_filepath(**modules)
    seq_info_path = _get_seqinfo_path(**modules)
    performance = _summerize_performance(evaluator_name, performance)

    print("Performance Metrics")
    if eval_encode_type == "bpp":
        avg_bpp = _calc_bpp(coded_res_df)
        result_df = pd.DataFrame({"avg_bpp": avg_bpp, "end_accuracy": performance})
        print(tabulate(result_df, headers="keys", tablefmt="psql"))

    if eval_encode_type == "bitrate":
        bitrate = _calc_bitrate(coded_res_df, seq_info_path)
        result_df = pd.DataFrame(
            {"bitrate (kbps)": bitrate, "end_accuracy": performance}
        )
        print(tabulate(result_df, headers="keys", tablefmt="psql"))

    print(f"Summary files saved in : {evaluator_filepath}")
    result_df.to_csv(
        os.path.join(evaluator_filepath, f'summary_{coded_res_df["qp"][0]}.csv'),
        index=False,
    )
    coded_res_df.to_csv(
        os.path.join(evaluator_filepath, f'encode_details_{coded_res_df["qp"][0]}.csv'),
        index=False,
    )


def _get_seq_info(seq_info_path):
    config = configparser.ConfigParser()
    config.read(seq_info_path)
    fps = config["Sequence"]["frameRate"]
    total_frame = config["Sequence"]["seqLength"]
    return int(fps), int(total_frame)


def _calc_bitrate(coded_res_df, seq_info_path):
    fps, total_frame = _get_seq_info(seq_info_path)
    print(f"Frame Rate: {fps}, Total Frame: {total_frame}")
    total_bytes = coded_res_df.groupby(["qp"])["bytes"].sum().tolist()[0]
    bitrate = ((total_bytes * 8) * fps) / (1000 * total_frame)
    return bitrate


def _calc_bpp(coded_res_df):
    total_bytes = coded_res_df.groupby(["qp"])["bytes"].sum().tolist()[0]
    total_pixels = coded_res_df.groupby(["qp"])["total_pixels"].sum().tolist()[0]
    avg_bpp = (total_bytes * 8) / total_pixels
    return avg_bpp


def _summerize_performance(evaluator_name, performance):
    if evaluator_name == "OpenImagesChallengeEval":
        value = [v for k, v in performance.items() if k.endswith("mAP@0.5IOU")]
        return value
    if evaluator_name == "MOT_TVD_Eval" or evaluator_name == "MOT_HiEve_Eval":
        value = [v for k, v in performance.items() if k == "mota"]
        return value
    return performance


def _get_evaluator_filepath(**modules):
    return modules["evaluator"].output_dir


def _get_evaluator_name(**modules):
    return title(modules["evaluator"])


def _get_seqinfo_path(**modules):
    return modules["dataloader"].dataset.seqinfo_path


if __name__ == "__main__":
    main()
