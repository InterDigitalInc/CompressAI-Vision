#!/usr/bin/env bash
#
# This clones and build model architectures and gets pretrained weights
set -eu

# default values, to be adapted w.r.t. MPEG FCVCM documents
TORCH_VERSION="2.0.0" # "1.10.2"
TORCHVISION_VERSION="0.15.1" # "0.11.3"
CUDA_VERSION=""
MODEL="detectron2"
CPU="False"
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

MODELS_DIR="${SCRIPT_DIR}/../models"
mkdir -p ${MODELS_DIR}

# Constrain DNNL to avoid AVX512, which leads to non-deterministic operation across different CPUs...
export DNNL_MAX_CPU_ISA=AVX2

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
                [--cuda) provide cuda version e.g. "11.8", default: check nvcc output)]
                [--detectron2_url use this if you want to specify a pre-built detectron2 (find at
                    "https://detectron2.readthedocs.io/en/latest/tutorials/install.html#install-pre-built-detectron2-linux-only"),
                    not required for regular versions derived from cuda and torch versions above.
                    default:"https://dl.fbaipublicfiles.com/detectron2/wheels/cu102/torch1.9/index.html"]


EXAMPLE         [bash install_models.sh -m detectron2 -t "1.9.1" --cuda "11.8" --compressai /path/to/compressai]
_EOF_
            exit;
            ;;
        -m|--model) shift; MODEL="$1"; shift; ;;
        -t|--torch) shift; TORCH_VERSION="$1"; shift; ;;
        --torchvision) shift; TORCHVISION_VERSION="$1"; shift; ;;
        --cpu) shift; CPU="True"; shift; ;;
        --cuda) shift; CUDA_VERSION="$1"; shift; ;;
        --detectron2_url) shift; DETECTRON2="$1"; shift; ;;
        *) echo "[ERROR] Unknown parameter $1"; exit; ;;
    esac;
done;

## Detectron2
if [ ${MODEL} == "detectron2" ] || [ ${MODEL} == "all" ]; then

    echo
    echo "Installing detectron2"
    echo

    # clone
    if [ -z "$(ls -A ${MODELS_DIR}/detectron2)" ]; then
        git clone https://github.com/facebookresearch/detectron2.git ${MODELS_DIR}/detectron2
    fi
    cd ${MODELS_DIR}/detectron2

    echo
    echo "checkout branch compatible with MPEG FCVCM"
    echo
    git -c advice.detachedHead=false  checkout 175b2453c2bc4227b8039118c01494ee75b08136

    if [ "${CUDA_VERSION}" == "" ] && [ "${CPU}" == "False" ]; then
        CUDA_VERSION=$(nvcc --version | sed -n 's/^.*release \([0-9]\+\.[0-9]\+\).*$/\1/p')
        if [ ${CUDA_VERSION} == "" ]; then
            echo "error with cuda, check your system, source env_cuda.sh or specify cuda version as argument."
        fi
    fi
    if [ -z "$CUDA_VERSION" ] || [ "$CPU" == "True" ]; then
        echo "installing on cpu"
        pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
    else
        echo "cuda version: $CUDA_VERSION"
        pip install torch==${TORCH_VERSION}+cu${CUDA_VERSION//./} torchvision==${TORCHVISION_VERSION}+cu${CUDA_VERSION//./} --index-url https://download.pytorch.org/whl/cu${CUDA_VERSION//./}
        wait
    fi
    pip install -e .

    # back to project root
    cd ${SCRIPT_DIR}/..


    echo
    echo "Downloading model weights"
    echo

    # FASTER R-CNN X-101 32x8d FPN
    WEIGHT_DIR="weights/detectron2/COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x/139173657"
    mkdir -p ${WEIGHT_DIR}
    wget -nc https://dl.fbaipublicfiles.com/detectron2/COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x/139173657/model_final_68b088.pkl -P ${WEIGHT_DIR}

    # FASTER R-CNN R-50 FPN
    WEIGHT_DIR="weights/detectron2/COCO-Detection/faster_rcnn_R_50_FPN_3x/137849458"
    mkdir -p ${WEIGHT_DIR}
    wget -nc https://dl.fbaipublicfiles.com/detectron2/COCO-Detection/faster_rcnn_R_50_FPN_3x/137849458/model_final_280758.pkl -P ${WEIGHT_DIR}

    # MASK R-CNN X-101 32x8d FPN
    WEIGHT_DIR="weights/detectron2/COCO-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_3x/139653917"
    mkdir -p ${WEIGHT_DIR}
    wget -nc https://dl.fbaipublicfiles.com/detectron2/COCO-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_3x/139653917/model_final_2d9806.pkl -P ${WEIGHT_DIR}

    # MASK R-CNN R-50 FPN
    WEIGHT_DIR="weights/detectron2/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x/137849600"
    mkdir -p ${WEIGHT_DIR}
    wget -nc https://dl.fbaipublicfiles.com/detectron2/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x/137849600/model_final_f10217.pkl -P ${WEIGHT_DIR}
fi

if [ ${MODEL} == "JDE" ] || [ ${MODEL} == "all" ]; then

    echo
    echo "Installing JDE"
    echo


    # install dependent packages
    pip3 install motmetrics numba lap opencv-python

    # install cython manually from source code with patch
    if [ -z "$(ls -A ${SCRIPT_DIR}/cython_bbox)" ]; then
        git clone https://github.com/samson-wang/cython_bbox.git ${SCRIPT_DIR}/cython_bbox
    fi

    cd ${SCRIPT_DIR}/cython_bbox
    patch -p1 --forward <../0001-compatible-with-numpy-1.24.1.patch
    pip3 install -e .

    # clone
    if [ -z "$(ls -A ${MODELS_DIR}/Towards-Realtime-MOT)" ]; then
        git clone https://github.com/Zhongdao/Towards-Realtime-MOT.git ${MODELS_DIR}/Towards-Realtime-MOT
    fi
    cd ${MODELS_DIR}/Towards-Realtime-MOT

    # git checkout
    echo
    echo "checkout branch compatible with MPEG FCVCM"
    echo
    git -c advice.detachedHead=false checkout c2654cdd7b69d39af669cff90758c04436025fe1

    # Apply patch to interface with compressai-vision
    patch -p1 --forward <${SCRIPT_DIR}/0001-interface-with-compressai-vision.patch

    # TODO (fabien - please check the script below)
    SITE_PACKAGES=$(python -c "import site; print(site.getsitepackages()[0])")
    # COPY JDE files into site-package under virtual environment
    if [ "${SITE_PACKAGES}" == "" ]; then
        echo "error with python site-packages directory, check your system and 'which python'"
        echo "ERROR: Fail to install JDE"
    fi
    
    mkdir -p ${SITE_PACKAGES}/jde
    cp models.py ${SITE_PACKAGES}/jde
    cp -r tracker ${SITE_PACKAGES}/jde/
    cp -r utils ${SITE_PACKAGES}/jde/
    echo "Complete copying jde files to site-packages @ ${SITE_PACKAGES}/jde"
    
    # back to project root
    cd ${SCRIPT_DIR}/..

    # download weights
    echo
    echo "Downloading weights..."
    echo

    WEIGHT_DIR=""weights/jde""
    mkdir -p ${WEIGHT_DIR}
    
    FILEID='1nlnuYfGNuHWZztQHXwVZSL_FvfE551pA'
    OUTFILE='jde.1088x608.uncertainty.pt'
    wget -nc --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id='${FILEID} -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=${FILEID}" -O ${WEIGHT_DIR}/${OUTFILE} && rm -rf /tmp/cookies.txt
fi


echo
echo "Installing compressai-vision"
echo

pip install -e .
