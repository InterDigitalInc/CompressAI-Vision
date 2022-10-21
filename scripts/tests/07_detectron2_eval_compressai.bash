#!/usr/bin/env bash
#
# tests an evaluation pipeline with a compression model from Compressai and
# object detection using detectron2
set -e

echo
echo 07
echo

compressai-vision detectron2-eval --y --dataset-name=mpeg-vcm-detection \
--slice=0:2 \
--scale=100 \
--progress=1 \
--qpars=1 \
--compressai-model-name=bmshj2018-factorized \
--output=detectron2_test.json \
--model=COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x.yaml
