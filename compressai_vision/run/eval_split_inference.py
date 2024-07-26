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
Runs and evaluates a split-inference pipeline

To evaluate the compression and accuracy performance in the split-inference pipeline, please run the following command:

.. code-block:: bash

    compressai-split-inference \
        --config-path="cfgs/eval_split_inference_example" 
        ...

Please check the scripts provided in scripts/evaluation for examples with supported codecs and datasets
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

import hydra
import pandas as pd
from omegaconf import DictConfig, OmegaConf
from tabulate import tabulate

from compressai_vision.datasets import get_seq_info
from compressai_vision.utils import get_max_num_cpus

thisdir = Path(__file__).parent
config_path = str(thisdir.joinpath("../../cfgs").resolve())


from compressai_vision.config import (
    configure_conf,
    create_codec,
    create_dataloader,
    create_evaluator,
    create_pipline,
    create_vision_model,
    write_outputs,
)


def setup(conf: DictConfig) -> dict[str, Any]:
    configure_conf(conf)

    vision_model = create_vision_model(conf.misc.device.nn_parts, conf.vision_model)
    dataloader = create_dataloader(
        conf.dataset, conf.misc.device.nn_parts, vision_model.cfg
    )
    evaluator = create_evaluator(
        conf.evaluator,
        conf.dataset.datacatalog,
        conf.dataset.config.dataset_name,
        dataloader.dataset,
    )

    if (
        Path(f"{conf.evaluator['output']}/summary.csv").is_file()
        and not conf.evaluator["overwrite_results"]
    ):
        print(
            "Corresponding summary.csv already exists and evaluator.overwrite_results is False, exiting..."
        )
        raise SystemExit(0)

    codec = create_codec(conf.codec, vision_model, conf.dataset)

    pipeline = create_pipline(conf.pipeline, conf.misc.device)

    write_outputs(conf)

    return pipeline, {
        "vision_model": vision_model,
        "codec": codec,
        "dataloader": dataloader,
        "evaluator": evaluator,
    }


title = lambda a: str(a.__class__).split("<class '")[-1].split("'>")[0].split(".")[-1]


def print_specs(pipeline, **kwargs):
    logger = logging.getLogger(__name__)

    # optional inputs
    annotation_path = "N/A"
    if kwargs["dataloader"].dataset.annotation_path:
        annotation_path = Path(kwargs["dataloader"].dataset.annotation_path).resolve()

    seqinfo_path = "N/A"
    if kwargs["dataloader"].dataset.seqinfo_path:
        seqinfo_path = Path(kwargs["dataloader"].dataset.seqinfo_path).resolve()

    info = f"\
                \n {'='*60}\
                \n Pipeline                   : {title(pipeline):<30s}\
                \n Vision Model               : {title(kwargs['vision_model']):<30s}\
                \n  -- Split Point            : {Path(kwargs['vision_model'].split_id).resolve()}\
                \n  -- Cfg                    : {Path(kwargs['vision_model'].model_cfg_path).resolve()}\
                \n  -- Weights                : {Path(kwargs['vision_model'].pretrained_weight_path).resolve()}\
                \n Codec                      : {title(kwargs['codec']):<30s}\
                \n  -- Counted # CPUs for use : {get_max_num_cpus()}\
                \n  -- Enc. Only              : {pipeline.configs['codec'].encode_only} \
                \n  -- Dec. Only              : {pipeline.configs['codec'].decode_only} \
                \n  -- Output Dir             : {Path(pipeline.codec_output_dir).resolve()} \
                \n  -- Skip N-Frames          : {pipeline.configs['codec'].skip_n_frames} \
                \n  -- # Frames To Be Coded   : {pipeline.configs['codec'].n_frames_to_be_encoded} \
                \n  -- Bitstream              : {pipeline.bitstream_name}.bin \
                \n Dataset                    : {kwargs['dataloader'].dataset.dataset_name} \
                \n  -- Data                   : {Path(kwargs['dataloader'].dataset.images_folder).resolve()} \
                \n  -- Annotation             : {annotation_path} \
                \n  -- SEQ-INFO               : {seqinfo_path} \
        "

    if kwargs["evaluator"]:
        info += f"\
                \n Evaluator                  : {title(kwargs['evaluator']):<30s}\
                \n  -- DataCatalog            : {kwargs['evaluator'].datacatalog_name} \
                \n  -- Output Dir             : {Path(kwargs['evaluator'].output_dir).resolve()} \
                \n  -- Output file            : {kwargs['evaluator'].output_file_name} \
        "
    info += f"\n\n"

    logger.info(info)


@hydra.main(version_base=None, config_path=str(config_path))
def main(conf: DictConfig):
    pipeline, modules = setup(conf)

    print_specs(pipeline, **modules)
    elap_times, eval_encode_type, coded_res, performance, mac_complexity = pipeline(
        **modules
    )

    if coded_res is not None:  # Encode Only
        # pretty output
        coded_res_df = pd.DataFrame(coded_res)

        print("=" * 100)
        print(f"Encoding Information [Top 5 Rows...][{pipeline}]")
        coded_res_df["file_name"] = coded_res_df["file_name"].apply(
            lambda x: Path(x).name
        )
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

    if modules["evaluator"] is None:
        print("[End of process] Terminated - No evaluation assigned\n")
        return

    # summarize results
    evaluator_name = _get_evaluator_name(**modules)
    evaluator_filepath = _get_evaluator_filepath(**modules)
    seq_info_path = _get_seqinfo_path(**modules)

    print(f"\nSummary files saved in : {evaluator_filepath}\n")
    if conf.pipeline["codec"]["encode_only"] is True:
        start_frm_idx = conf.pipeline["codec"]["skip_n_frames"]
        end_frm_idx = start_frm_idx + conf.pipeline["codec"]["n_frames_to_be_encoded"]

        result_df = pd.DataFrame(
            {
                "start_frm_idx": start_frm_idx,
                "end_frm_idx": end_frm_idx,
                **elap_times,
            },
            index=[0],
        )
        print(tabulate(result_df, headers="keys", tablefmt="psql"))

        result_df.to_csv(
            os.path.join(
                evaluator_filepath,
                f"summary_encode_only_between_{start_frm_idx:05d}_and_{end_frm_idx:05d}.csv",
            ),
            index=False,
        )
        print("[End of process] Terminated - Encoding only has finished\n")
        return

    if conf.pipeline["codec"]["decode_only"] is True:
        enc_summary = pd.DataFrame()
        for file_path in Path(evaluator_filepath).glob(
            "summary_encode_only_between_*.csv"
        ):
            df = pd.read_csv(file_path)
            enc_summary = pd.concat([enc_summary, df])

        if len(enc_summary) > 0:
            # [TODO] should be cleaned up
            elap_times["nn_part_1"] = enc_summary["nn_part_1"].sum()

            tmp_dict = {}
            for k in enc_summary:
                if "encode" in k:
                    tmp_dict[k] = enc_summary[k].sum()

            updates = {}
            # add time details after encode
            for k, v in elap_times.items():
                updates[k] = v
                if k == "encode":
                    for sk, sv in tmp_dict.items():
                        updates[sk] = sv

            elap_times = updates

    performance, eval_criteria = _summerize_performance(
        evaluator_name, performance, conf.evaluator.eval_criteria
    )

    print(f"\nPerformance Metrics Using Evaluation Criteria {eval_criteria}\n")
    if eval_encode_type == "bpp":
        dataset_name = _get_dataset_name(**modules)
        avg_bpp = _calc_bpp(coded_res_df)
        result_df = pd.DataFrame(
            {
                "Dataset": dataset_name,
                "qp": coded_res_df["qp"][0],
                "avg_bpp": avg_bpp,
                "end_accuracy": performance,
                **elap_times,
            }
        )
        print(tabulate(result_df, headers="keys", tablefmt="psql"))

    if eval_encode_type == "bitrate":
        name, fps, total_frame, bitrate = _calc_bitrate(coded_res_df, seq_info_path)
        result_df = pd.DataFrame(
            {
                "Dataset": name,
                "fps": fps,
                "num_of_coded_frame": total_frame,
                "qp": coded_res_df["qp"][0],
                "bitrate (kbps)": bitrate,
                "end_accuracy": performance,
                **elap_times,
            }
        )
        print(tabulate(result_df, headers="keys", tablefmt="psql"))

    if conf.codec["mac_computation"]:
        calc_mac_df = pd.DataFrame(
            {
                "Metric": "KMAC/pixel",
                "nn_part1": mac_complexity["nn_part_1"],
                "feature reduction": mac_complexity["feature_reduction"],
                "feature restoration": mac_complexity["feature_restoration"],
                "nn_part2": mac_complexity["nn_part_2"],
            },
            index=[0],
        )
        print("Complexity Measurement (KMAC/pixel)")
        print(tabulate(calc_mac_df, headers="keys", tablefmt="psql"))

        calc_mac_df.to_csv(
            os.path.join(evaluator_filepath, f"summary_complexity.csv"),
            index=False,
        )

    result_df.to_csv(
        os.path.join(evaluator_filepath, f"summary.csv"),
        index=False,
    )

    coded_res_df.to_csv(
        os.path.join(evaluator_filepath, f'encode_details_{coded_res_df["qp"][0]}.csv'),
        index=False,
    )


def _calc_bitrate(coded_res_df, seq_info_path):
    name, fps, total_frame = get_seq_info(seq_info_path)
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
        value = [v for k, v in performance["bbox"].items() if k == eval_criteria]
        if not value:
            print(
                f"\n{eval_criteria} is not supported for {evaluator_name}, using default evaluation criteria {def_criteria}"
            )
            eval_criteria = def_criteria
            value = [v for k, v in performance["bbox"].items() if k == eval_criteria]
        return value, eval_criteria

    if evaluator_name == "MOT_TVD_Eval" or evaluator_name == "MOT_HiEve_Eval":
        def_criteria = "mota"
        if not eval_criteria:
            eval_criteria = def_criteria
        value = [v for k, v in performance.items() if k == eval_criteria]
        if not value:
            print(
                f"\n{eval_criteria} is not supported for {evaluator_name}, using default evaluation criteria {def_criteria}"
            )
            eval_criteria = def_criteria
            value = [v for k, v in performance.items() if k == eval_criteria]
        return value, eval_criteria
    return performance, eval_criteria


def _get_evaluator_filepath(**modules):
    return modules["evaluator"].output_dir


def _get_evaluator_name(**modules):
    return title(modules["evaluator"])


def _get_dataset_name(**modules):
    return modules["evaluator"].dataset_name


def _get_seqinfo_path(**modules):
    return modules["dataloader"].dataset.seqinfo_path


if __name__ == "__main__":
    main()
