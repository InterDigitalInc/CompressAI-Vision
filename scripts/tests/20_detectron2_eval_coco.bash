#!/usr/bin/env bash
# tests inferring detectron2 without compression of the input images before

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

echo
echo 11
echo

compressai-vision detectron2-eval --y --dataset-name=coco2017-eval \
--slice=0:2 \
--scale=100 \
--progress=1 \
--output=detectron2_test.json \
--model=COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x.yaml \
--gt-field=ground_truthx \
--eval-method=coco
