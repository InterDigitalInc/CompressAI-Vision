#!/usr/bin/env bash
#
# This clones and build model architectures and gets pretrained weights
set -eu

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

MODELS_DIR="${SCRIPT_DIR}/../models"
mkdir -p ${MODELS_DIR}

## Detectron2

# clone
if [ -z "$(ls -A ${MODELS_DIR}/detectron2)" ]; then
    git clone https://github.com/facebookresearch/detectron2.git ${MODELS_DIR}/detectron2
fi
cd ${MODELS_DIR}/detectron2

# to be compatible with MPEG FCVCM
git checkout 175b2453c2bc4227b8039118c01494ee75b08136

# pip install torch==2.0.0+cu118 torchvision==0.15.1+cu118 torchaudio==2.0.1 --index-url https://download.pytorch.org/whl/cu118
# is there a way this script capture the current nvcc version and depending on
# the version, propose a specific version torch installation?

TORCH_VERSION="2.0.0"
CUDA_VERSION=$(nvcc --version | sed -n 's/^.*release \([0-9]\+\.[0-9]\+\).*$/\1/p')
if [ -z "$CUDA_VERSION" ]; then
    echo "no CUDA detected, installing on cpu"
    pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
else
    echo "cuda version: $CUDA_VERSION"
    python3 -m pip install -U torch==${TORCH_VERSION}+cu${CUDA_VERSION//./}$ torchvision==0.15.1+cu${CUDA_VERSION//./} torchaudio==2.0.1 --index-url https://download.pytorch.org/whl/cu${CUDA_VERSION//./}
fi

python3 -m pip install .

cd ../../

#downaload weights
WEIGHT_DIR="weights/detectron2/COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x/139173657"
mkdir -p ${WEIGHT_DIR}
wget https://dl.fbaipublicfiles.com/detectron2/COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x/139173657/model_final_68b088.pkl \
 ${WEIGHT_DIR}



## JDE
# mkdir -p "${SCRIPT_DIR}/../models/jde"
