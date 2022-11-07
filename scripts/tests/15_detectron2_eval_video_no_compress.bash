#!/usr/bin/env bash
# tests inferring detectron2 without compression of the input images before
# video version

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

echo
echo 15
echo

# --progress=1 \
compressai-vision detectron2-eval --y --dataset-name=sfu-dummy \
--scale=100 \
--progressbar \
--output=detectron2_test.json \
--model=COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x.yaml
