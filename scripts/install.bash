#!/usr/bin/env bash
#
# This will install compressai-vision using the provided versions of the
# dependencies within a virtual environment with python3 and up-to-date pip
set -eu

TORCH="1.9.1"
TORCHVISION="0.10.1"
CUDA="cu102"
DETECTRON2="https://dl.fbaipublicfiles.com/detectron2/wheels/${CUDA}/torch${TORCH::-2}/index.html"
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
COMPRESSAI=""
# WARNING: keep consistent with ../compressai_vision/__init__.py:
FO_VERSION="0.16.6" 

#test python version>3.8
python3 -c "import sys; assert(sys.version_info.major>=3); assert(sys.version_info.minor>=8)"
if [[ $? -gt 0 ]]
then
    echo
    echo "Your python version needs to be >=3.8"
    echo
    exit 2
fi

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

RUN OPTIONS:    [-t|--torch torch version, default="1.9.1"]
                [-v|--torchvision torchvision version, default="0.10.1"]
                [--detectron2_url location of proper pre-built detectron2 (find at
                    "https://detectron2.readthedocs.io/en/latest/tutorials/install.html#install-pre-built-detectron2-linux-only"),
                    if not provided, cuda and="https://dl.fbaipublicfiles.com/detectron2/wheels/cu102/torch1.9/index.html"]
                [--compressai) provide path to compressai source code for import (in editable mode), default: install compressai from PyPI)]
                [--cuda) provide cuda version in the form cu11.1 for cuda 11.1, or "cpu", default: "cu102")]

EXAMPLE         [bash install.sh -t "1.9.1" --cuda "cu102" --compressai /path/to/compressai]
_EOF_
            exit;
            ;;
        -t|--torch) shift; TORCH="$1"; shift; ;;
        -v|--torchvision) shift; TORCHVISION="$1"; shift; ;;
        --detectron2_url) shift; DETECTRON2="$1"; shift; ;;
        -c|--compressai) shift; COMPRESSAI="$1"; shift; ;;
        --cuda) shift; CUDA="$1"; shift; ;;
        *) echo "[ERROR] Unknown parameter $1"; exit; ;;
    esac;
done;


pip install -U pip
pip install fiftyone==$FO_VERSION jupyter ipython
pip install torch==${TORCH} torchvision==${TORCHVISION} pytorch-msssim
pip install detectron2 -f ${DETECTRON2}

pip install pybind11
if [ "${COMPRESSAI}" = "" ]; then
    pip install compressai
else
    pip install -e ${COMPRESSAI}
fi

# fix the virtualenv to include a mongodb name for fiftyone
python3 ${SCRIPT_DIR}/insert_venv.py

# install compressai-vision
pip install -e ${SCRIPT_DIR}/..
