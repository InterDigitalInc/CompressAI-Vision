#!/usr/bin/env bash
#
# tests an evaluation pipeline with a compression model from Compressai and
# object detection using detectron2
set -e

echo
echo 12
echo

# --progress=1 \
# --progressbar \
compressai-vision metrics-eval --y --dataset-name=oiv6-mpeg-detection-v1 \
--slice=0:2 \
--scale=100 \
--progress=1 \
--qpars=1 \
--compressai-model-name=bmshj2018_factorized \
--output=detectron2_metrics_test.json
