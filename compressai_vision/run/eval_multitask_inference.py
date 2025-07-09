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

r"""
Evaluate a system performance of end-to-end pipeline.



"""

from __future__ import annotations

from compressai_vision.config import (
    configure_conf,
    create_dataloader,
    create_evaluator,
    create_multi_task_codec,
    create_pipline,
    create_vision_model,
    write_outputs,
)

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
config_path = str(thisdir.joinpath("../../cfgs").resolve())


def setup(conf: DictConfig) -> dict[str, Any]:
    configure_conf(conf)

    # would there be any better way to do this?
    mtasks = [
        (key, item)
        for key, item in conf.codec.decoder_config.items()
        if "tlayer" in key and isinstance(item, DictConfig)
    ]
    mtasks = list(dict(sorted(mtasks, key=lambda x: x[0])).values())

    mevals = [
        (key, item)
        for key, item in conf.evaluator.items()
        if "tlayer" in key and isinstance(item, DictConfig)
    ]
    mevals = list(dict(sorted(mevals, key=lambda x: x[0])).values())

    # vision network configuration needed to properly scale input for dataloader (in case measure machine performance)
    trg_arch = mtasks[conf.pipeline.codec.target_task_layer].vision_arch

    vision_cfg = None
    if trg_arch.lower() != "none":
        conf.vision_model.arch = trg_arch
        vision_cfg = create_vision_model(conf.misc.device, conf.vision_model).cfg

    dataloader = create_dataloader(conf.dataset, conf.misc.device, vision_cfg)

    vision_models = []
    evaluators = []

    for e, task_eval in enumerate(zip(mtasks, mevals)):
        task, teval = task_eval

        conf.vision_model.arch = task["vision_arch"]
        vm = create_vision_model(conf.misc.device, conf.vision_model)

        if e == conf.pipeline.codec.target_task_layer:
            eval = create_evaluator(
                teval,
                conf.dataset.datacatalog,
                conf.dataset.config.dataset_name,
                dataloader.dataset,
            )
        else:
            eval = None

        vision_models.append(vm)
        evaluators.append(eval)

    codec = create_multi_task_codec(conf.codec, vision_models, conf.misc.device)

    pipeline = create_pipline(conf.pipeline, conf.misc.device)

    write_outputs(conf)

    return pipeline, {
        "codec": codec,
        "vision_models": vision_models,
        "dataloader": dataloader,
        "evaluators": evaluators,
    }


def title(a):
    return str(a.__class__).split("<class '")[-1].split("'>")[0].split(".")[-1]


def print_specs(pipeline, **kwargs):
    logger = logging.getLogger(__name__)

    log_str = f"\n {'='*60}"
    log_str += f"\n Pipeline                 : {title(pipeline):<30s}"
    log_str += f"\n Codec                    : {title(kwargs['codec']):<30s}"
    log_str += f"\n  -- Enc. Only            : {pipeline.configs['codec'].encode_only}"
    log_str += f"\n  -- Dec. Only            : {pipeline.configs['codec'].decode_only}"
    log_str += (
        f"\n  -- Output Dir           : {Path(pipeline.codec_output_dir).resolve()}"
    )
    log_str += (
        f"\n  -- Skip N-Frames        : {pipeline.configs['codec'].skip_n_frames}"
    )
    log_str += f"\n  -- # Frames To Be Coded : {pipeline.configs['codec'].n_frames_to_be_encoded}"
    log_str += f"\n  -- Bitstream            : {pipeline.bitstream_name}.bin"

    if "num_tasks" in pipeline.configs["codec"]:
        log_str += (
            f"\n  -- # of tasks           : {pipeline.configs['codec'].num_tasks}"
        )
    if "target_task_layer" in pipeline.configs["codec"]:
        log_str += f"\n  -- Target task layer    : {pipeline.configs['codec'].target_task_layer}"

    log_str += "\n Multiple Tasks"
    for e, task in enumerate(kwargs["vision_models"]):
        if task is not None:
            log_str += f"\n  -- L{e} Vision Model      : {title(task)}"
            log_str += (
                f"\n     -- Cfg               : {Path(task.model_cfg_path).resolve()}"
            )
            log_str += f"\n     -- Weight            : {Path(task.pretrained_weight_path).resolve()}"
        else:
            log_str += f"\n  -- L{e} Human Visual"

    log_str += (
        f"\n Dataset                  : {kwargs['dataloader'].dataset.dataset_name}"
    )
    log_str += f"\n  -- Data                 : {Path(kwargs['dataloader'].dataset.images_folder).resolve()}"
    if kwargs["dataloader"].dataset.annotation_path is not None:
        log_str += f"\n  -- Annotation           : {Path(kwargs['dataloader'].dataset.annotation_path).resolve()}"
    if kwargs["dataloader"].dataset.seqinfo_path is not None:
        log_str += f"\n  -- SEQ-INFO             : {Path(kwargs['dataloader'].dataset.seqinfo_path).resolve()}"
    log_str += "\n Evaluators"

    assert "target_task_layer" in pipeline.configs["codec"]

    for e, teval in enumerate(kwargs["evaluators"]):
        if e == pipeline.configs["codec"].target_task_layer:
            log_str += f"\n  -- L{e} Type              : {title(teval)} "
            log_str += f"\n     -- DataCatalog       : {teval.datacatalog_name} "
            log_str += (
                f"\n     -- Output Dir        : {Path(teval.output_dir).resolve()}"
            )
            log_str += f"\n     -- Output file       : {teval.output_file_name}"

    log_str += "\n\n"

    logger.info(log_str)


@hydra.main(version_base=None, config_path=str(config_path))
def main(conf: DictConfig):
    pipeline, modules = setup(conf)

    print_specs(pipeline, **modules)
    timing, eval_encode_type, coded_res, performance = pipeline(**modules)

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

    # summarize results
    tlid = pipeline.configs["codec"].target_task_layer
    evaluator_name = _get_evaluator_name(modules["evaluators"][tlid])
    evaluator_filepath = _get_evaluator_filepath(modules["evaluators"][tlid])
    # seq_info_path = _get_seqinfo_path(**modules)
    performance, eval_criteria = _summerize_performance(
        evaluator_name, performance, modules["evaluators"][tlid].criteria
    )

    print("\nPerformance Metrics\n")
    assert eval_encode_type == "bpp"
    if eval_encode_type == "bpp":
        dataset_name = _get_dataset_name(modules["evaluators"][tlid])
        avg_bpp = _calc_bpp(coded_res_df)
        result_df = pd.DataFrame(
            {
                "Dataset": dataset_name,
                "qp": coded_res_df["qp"][0],
                "avg_bpp": avg_bpp,
                "end_accuracy": performance,
                **timing,
            }
        )
        print(tabulate(result_df, headers="keys", tablefmt="psql"))

    print(f"\nSummary files saved in : {evaluator_filepath}\n")
    result_df.to_csv(
        os.path.join(evaluator_filepath, "summary.csv"),
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
    name = f'{config["Sequence"]["name"]}_{config["Sequence"]["imWidth"]}x{config["Sequence"]["imHeight"]}_{fps}'
    return name, int(fps), int(total_frame)


def _calc_bitrate(coded_res_df, seq_info_path):
    name, fps, total_frame = _get_seq_info(seq_info_path)
    print(f"Frame Rate: {fps}, Total Frame: {total_frame}")
    total_bytes = coded_res_df.groupby(["qp"])["bytes"].sum().tolist()[0]
    bitrate = ((total_bytes * 8) * fps) / (1000 * total_frame)
    return name, fps, total_frame, bitrate


def _calc_bpp(coded_res_df):
    total_bytes = coded_res_df.groupby(["qp"])["bytes"].sum().tolist()[0]
    total_pixels = coded_res_df.groupby(["qp"])["total_pixels"].sum().tolist()[0]
    avg_bpp = (total_bytes * 8) / total_pixels
    return avg_bpp


def _summerize_performance(evaluator_name, performance, eval_criteria):
    if evaluator_name == "OpenImagesChallengeEval":
        def_criteria = "mAP@0.5IOU"
        if not eval_criteria:
            eval_criteria = def_criteria
        value = [v for k, v in performance.items() if k.endswith(eval_criteria)]
        if not value:
            print(
                f"\n{eval_criteria} is not supported for {evaluator_name}, using default evaluation criteria {def_criteria}"
            )
            eval_criteria = def_criteria
            value = [v for k, v in performance.items() if k.endswith(eval_criteria)]
        return value, eval_criteria

    if evaluator_name == "COCOEVal":
        def_criteria = "AP"
        if not eval_criteria:
            eval_criteria = def_criteria
        value = [
            v
            for k, v in performance["bbox"].items()
            if k.lower() == eval_criteria.lower()
        ]
        if not value:
            print(
                f"\n{eval_criteria} is not supported for {evaluator_name}, using default evaluation criteria {def_criteria}"
            )
            eval_criteria = def_criteria
            value = [
                v
                for k, v in performance["bbox"].items()
                if k.lower() == eval_criteria.lower()
            ]
        return value, eval_criteria

    if evaluator_name == "MOT_TVD_Eval" or evaluator_name == "MOT_HiEve_Eval":
        def_criteria = "mota"
        if not eval_criteria:
            eval_criteria = def_criteria
        value = [
            v for k, v in performance.items() if k.lower() == eval_criteria.lower()
        ]
        if not value:
            print(
                f"\n{eval_criteria} is not supported for {evaluator_name}, using default evaluation criteria {def_criteria}"
            )
            eval_criteria = def_criteria
            value = [
                v for k, v in performance.items() if k.lower() == eval_criteria.lower()
            ]
        return value, eval_criteria

    if evaluator_name == "VisualQuality":
        def_criteria = "psnr"
        if not eval_criteria:
            eval_criteria = def_criteria

        value = [
            v for k, v in performance.items() if k.lower() == eval_criteria.lower()
        ]
        if not value:
            print(
                f"\n{eval_criteria} is not supported for {evaluator_name}, using default evaluation criteria {def_criteria}"
            )
            eval_criteria = def_criteria
            value = [
                v for k, v in performance.items() if k.lower() == eval_criteria.lower()
            ]
        return value, eval_criteria

    return performance, eval_criteria


def _get_evaluator_filepath(evaluator):
    return evaluator.output_dir


def _get_evaluator_name(evaluator):
    return title(evaluator)


def _get_dataset_name(evaluator):
    return evaluator.dataset_name


def _get_seqinfo_path(dataloader):
    return dataloader.dataset.seqinfo_path


if __name__ == "__main__":
    main()
