#!/usr/bin/env bash
#
# This clones and build model architectures and gets pretrained weights
set -eu

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

mkdir -p "${SCRIPT_DIR}/../models"

## Detectron2

# clone
if [ -z "$(ls -A models/detectron2)" ]; then
    git clone https://github.com/facebookresearch/detectron2.git models/detectron2
fi
cd models/detectron2

# to be compatible with MPEG FCVCM
git checkout 175b2453c2bc4227b8039118c01494ee75b08136

# pip install torch==2.0.0+cu118 torchvision==0.15.1+cu118 torchaudio==2.0.1 --index-url https://download.pytorch.org/whl/cu118
# is there a way this script capture the current nvcc version and depending on
# the version, propose a specific version torch installation?
python3 -m pip install -U torch
python3 -m pip install .

#downaload weights
mkdir -p "detectron2/COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x/139173657"
wget https://dl.fbaipublicfiles.com/detectron2/COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x/139173657/model_final_68b088.pkl \
 weights/COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x/139173657

cd ../../

## JDE
# mkdir -p "${SCRIPT_DIR}/../models/jde"
