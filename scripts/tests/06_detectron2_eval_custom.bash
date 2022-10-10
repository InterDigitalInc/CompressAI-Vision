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
CKPT1=""
CKPT2=""

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

DEFAULT_CODEC="${SCRIPT_DIR}/../../examples/models/bmshj2018-factorized"
CKPT1="${DEFAULT_CODEC}/bmshj2018-factorized-prior-1-446d5c7f.pth.tar"
CKPT2="${DEFAULT_CODEC}/bmshj2018-factorized-prior-2-87279a02.pth.tar"

#parse args
while [[ $# -gt 0 ]]
do
    key="$1"
    case $key in
        -h|--help)
cat << _EOF_
OPTIONS: [--codec folder containing compression model for test, default="examples/models/bmshj2018-factorized"]
         [--ckpt1 folder containing a checkpoint for test, default="bmshj2018-factorized-prior-1-446d5c7f.pth.tar"]
         [--ckpt2 folder containing a checkpoint for test, default="bmshj2018-factorized-prior-2-87279a02.pth.tar"]
_EOF_
        exit;
        ;;
        --codec) shift; CODEC="$1"; shift; ;;
        --ckpt1) shift; CKPT1="$1"; shift; ;;
        --ckpt2) shift; CKPT2="$1"; shift; ;;
        *) echo "[ERROR] Unknown parameter $1"; exit; ;;
    esac;
done;

if [ "${CODEC}" = "" ]; then
    echo "using default examples/models/bmshj2018-factorized"
    CODEC=${DEFAULT_CODEC}
fi

# if [ "${CKPT1}" = "" ] || [ "${CKPT2}" = "" ]; then
#     echo "enter paths for codec folder, ckpt1 and ckpt2";
#     exit 1;
# fi

compressai-vision \
detectron2-eval \
--y \
--dataset-name mpeg-vcm-detection \
--slice 0:2 \
--scale 100 \
--progress 1 \
--compression-model-path $CODEC \
--compression-model-checkpoint $CKPT1 $CKPT2 \
--output=detectron2_bmshj2018-factorized.json \
--model=COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x.yaml