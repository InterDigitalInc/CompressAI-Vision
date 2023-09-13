#! /usr/bin/env bash

DATASET=$1
RESULT_PATH=$2

python ../../utils/mpeg_template_format.py --dataset ${DATASET} --result_path ${RESULT_PATH}
