#!/usr/bin/env bash
#
# This runs encode/decode using a custom codec and tuns detectron2
# inference on the decoded content
# It requires having imported and registered mpeg-vcm-detection
# i.e. run -1_auto_import_mock.bash
set -e

echo
echo 06
echo

CODEC=""


SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

DEFAULT_CODEC="${SCRIPT_DIR}/../../examples/models/bmshj2018-factorized"

CODEC="$1"
if [ "${CODEC}" = "" ]; then
    echo "using default examples/models/bmshj2018-factorized"
    CODEC=${DEFAULT_CODEC}
fi

compressai-vision \
detectron2-eval \
--y \
--dataset-name mpeg-vcm-detection \
--slice 0:2 \
--scale 100 \
--progress 1 \
--compression-model-path ${CODEC} \
--qpars=1 \
--output=detectron2_bmshj2018-factorized.json \
--model=COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x.yaml
