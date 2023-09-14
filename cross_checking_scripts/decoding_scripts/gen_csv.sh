#! /usr/bin/env bash

RUN="sequential" # "gnu_parallel" or "sequential" or "slurm"

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

DATASET=$1
RESULT_PATH=$2

python ${SCRIPT_DIR}/../../utils/mpeg_template_format.py --dataset ${DATASET} --result_path ${RESULT_PATH}
