#!/usr/bin/env bash
#
# This clones and build model architectures and gets pretrained weights
set -eu

# default values taken from MPEG FCVCM documents
TORCH_VERSION="1.10.2" # "2.0.1"
TORCHVISION_VERSION="0.11.3" # "1.15.1"
TORCHAUDIO_VERSION="0.10.2" # "2.0.2"
CUDA_VERSION=""
MODEL="detectron2"
CPU="False"
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

MODELS_DIR="${SCRIPT_DIR}/../models"
mkdir -p ${MODELS_DIR}


#parse args
while [[ $# -gt 0 ]]
do
    key="$1"
    case $key in
        -h|--help)
cat << _EOF_
Installs CompressAI-Vision and its dependencies within a virtual environment.
Before running, create a virtual env, i.e.:
$ python3 -m venv venv
$ source venv/bin/activate

RUN OPTIONS:
                [-m|--model, default=detectron2]
                [-t|--torch torch version, default="0.10.1"]
                [--torchvision torchvision version]
                [--torchaudio torchaudio version]
                [--cpu) build for cpu only)]
                [--cuda_version) provide cuda version e.g. "11.8", default: check nvcc output)]
                [--detectron2_url use this if you want to specify a pre-built detectron2 (find at
                    "https://detectron2.readthedocs.io/en/latest/tutorials/install.html#install-pre-built-detectron2-linux-only"),
                    not required for regular versions derived from cuda and torch versions above.
                    default:"https://dl.fbaipublicfiles.com/detectron2/wheels/cu102/torch1.9/index.html"]


EXAMPLE         [bash get_models.sh -m detectron2 -t "1.9.1" --cuda_version "11.8" --compressai /path/to/compressai]
_EOF_
            exit;
            ;;
        -m|--model) shift; MODEL="$1"; shift; ;;
        -t|--torch) shift; TORCH_VERSION="$1"; shift; ;;
        --torchvision) shift; TORCHVISION_VERSION="$1"; shift; ;;
        --torchaudio) shift; TORCHAUDIO_VERSION="$1"; shift; ;;
        --cpu) shift; CPU="True"; shift; ;;
        --cuda_version) shift; CUDA_VERSION="$1"; shift; ;;
        --detectron2_url) shift; DETECTRON2="$1"; shift; ;;
        *) echo "[ERROR] Unknown parameter $1"; exit; ;;
    esac;
done;


## Detectron2
if [ ${MODEL} == "detectron2" ] || [ ${MODEL} == "all" ]; then

    echo
    echo "Install detectron2"
    echo

    # clone
    if [ -z "$(ls -A ${MODELS_DIR}/detectron2)" ]; then
        git clone https://github.com/facebookresearch/detectron2.git ${MODELS_DIR}/detectron2
    fi
    cd ${MODELS_DIR}/detectron2

    # to be compatible with MPEG FCVCM
    git checkout 175b2453c2bc4227b8039118c01494ee75b08136

    if [ CUDA_VERSION == "" ] && [ $CPU == "False" ]; then
        CUDA_VERSION=$(nvcc --version | sed -n 's/^.*release \([0-9]\+\.[0-9]\+\).*$/\1/p')
        if [ CUDA_VERSION == "" ]; then
            echo "error with cuda, check your system or specify cuda version as argument."
        fi
    fi
    if [ -z "$CUDA_VERSION" ] || [ $CPU == "True" ]; then
        echo "installing on cpu"
        pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
    else
        echo "cuda version: $CUDA_VERSION"
        pip install torch==${TORCH_VERSION}+cu${CUDA_VERSION//./} torchvision==${TORCHVISION_VERSION}+cu${CUDA_VERSION//./} torchaudio==${TORCHAUDIO_VERSION}+cu${CUDA_VERSION//./} --index-url https://download.pytorch.org/whl/cu${CUDA_VERSION//./}
    fi

    pip install .

    cd ../../

    #download weights

    echo
    echo "Downloading weights"
    echo

    # FASTER R-CNN
    WEIGHT_DIR="weights/detectron2/COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x/139173657"
    mkdir -p ${WEIGHT_DIR}
    wget -nc https://dl.fbaipublicfiles.com/detectron2/COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x/139173657/model_final_68b088.pkl -P ${WEIGHT_DIR}

    # MASK R-CNN
    WEIGHT_DIR="weights/detectron2/COCO-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_3x/139653917"
    mkdir -p ${WEIGHT_DIR}
    wget -nc https://dl.fbaipublicfiles.com/detectron2/COCO-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_3x/139653917/model_final_2d9806.pkl -P ${WEIGHT_DIR}
fi


## JDE
# mkdir -p "${SCRIPT_DIR}/../models/jde"
