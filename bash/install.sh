#!/usr/bin/env bash
#
# This will install compressai-vision using the provided versions of the
# dependencies within a virtual environment with python3 and up-to-date pip 
set -eu

TORCH="1.9.1"
TORCHVISION="0.10.1"
DETECTRON2="https://dl.fbaipublicfiles.com/detectron2/wheels/cu102/torch1.9/index.html"
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

while [[ $# -gt 0 ]]
do
    key="$1"
    case $key in
        -h|--help)
cat << _EOF_
RUN OPTIONS:    [-t|--torch torch version, default="1.9.1"]
                [-v|--torchvision torchvision version, default="0.10.1"]
                [-d|--detectron2 location of proper pre-built detectron2 (find at "https://detectron2.readthedocs.io/en/latest/tutorials/install.html#install-pre-built-detectron2-linux-only"), default="https://dl.fbaipublicfiles.com/detectron2/wheels/cu102/torch1.9/index.html"]

EXAMPLE         [bash install.sh -t "1.9.1" -d https://dl.fbaipublicfiles.com/detectron2/wheels/cpu/torch1.9/index.html]
_EOF_
            exit;
            ;;
        -t|--torch) shift; TORCH="$1"; shift; ;;
        -v|--torchvision) shift; TORCHVISION="$1"; shift; ;;
        -d|--detectron2) shift; DETECTRON2="$1"; shift; ;;
        *) echo "[ERROR] Unknown parameter $1"; exit; ;;
    esac;
done;

pip install -U pip
pip install fiftyone jupyter
pip install torch==${TORCH} torchvision==${TORCHVISION} pytorch-msssim
pip install detectron2 -f ${DETECTRON2}
pip install compressai
## WARNING TODO: this wont work until CompressAI-Vision becomes all public:
pip install -e ${SCRIPT_DIR}/..
