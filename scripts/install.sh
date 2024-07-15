#!/usr/bin/env bash
#
# This clones and build model architectures and gets pretrained weights
set -eu

TORCH_VERSION="2.0.0"
TORCHVISION_VERSION="0.15.1"
CUDA_VERSION=""
MODEL="all"
CPU="False"
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
MODELS_ROOT_DIR="${SCRIPT_DIR}/.."

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
                [-m|--model, default=all]
                [-t|--torch torch version, default="2.0.0"]
                [--torchvision torchvision version, default="0.15.1"]
                [--cpu) build for cpu only)]
                [--cuda) provide cuda version e.g. "11.8", default: check nvcc output)]
                [--detectron2_url use this if you want to specify a pre-built detectron2 (find at
                    "https://detectron2.readthedocs.io/en/latest/tutorials/install.html#install-pre-built-detectron2-linux-only"),
                    not required for regular versions derived from cuda and torch versions above.
                    default:"https://dl.fbaipublicfiles.com/detectron2/wheels/cu102/torch1.9/index.html"]
                [--models_dir directory to install vision models to, default: compressai_vision_root]


EXAMPLE         [bash install_models.sh -m detectron2 -t "1.9.1" --cuda "11.8" --compressai /path/to/compressai]

NOTE: the downlading of JDE pretrained weights might fail. Check that the size of following file is ~558MB.
compressai_vision/weights/jde/jde.1088x608.uncertainty.pt
The file can be downloaded at the following link (in place of the above file path):
"https://docs.google.com/uc?export=download&id=1nlnuYfGNuHWZztQHXwVZSL_FvfE551pA"
_EOF_
            exit;
            ;;
        -m|--model) shift; MODEL="$1"; shift; ;;
        -t|--torch) shift; TORCH_VERSION="$1"; shift; ;;
        --torchvision) shift; TORCHVISION_VERSION="$1"; shift; ;;
        --cpu) CPU="True"; shift; ;;
        --cuda) shift; CUDA_VERSION="$1"; shift; ;;
        --detectron2_url) shift; DETECTRON2="$1"; shift; ;;
        --models_dir) shift; MODELS_ROOT_DIR="$1"; shift; ;;
        *) echo "[ERROR] Unknown parameter $1"; exit; ;;
    esac;
done;

MODELS_SOURCE_DIR=${MODELS_ROOT_DIR}/models
MODELS_WEIGHT_DIR=${MODELS_ROOT_DIR}/weights

mkdir -p ${MODELS_SOURCE_DIR}
mkdir -p ${MODELS_WEIGHT_DIR}

## Make sure we have up-to-date pip and wheel
pip3 install -U pip wheel

## Detectron2
if [ ${MODEL} == "detectron2" ] || [ ${MODEL} == "all" ]; then

    echo
    echo "Installing detectron2"
    echo

    # clone
    if [ -z "$(ls -A ${MODELS_SOURCE_DIR}/detectron2)" ]; then
        git clone https://github.com/facebookresearch/detectron2.git ${MODELS_SOURCE_DIR}/detectron2
    fi
    cd ${MODELS_SOURCE_DIR}/detectron2

    echo
    echo "checkout the version used for MPEG FCM"
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
        wait
    else
        echo "cuda version: $CUDA_VERSION"
        pip3 install torch==${TORCH_VERSION}+cu${CUDA_VERSION//./} torchvision==${TORCHVISION_VERSION}+cu${CUDA_VERSION//./} --extra-index-url https://download.pytorch.org/whl/cu${CUDA_VERSION//./}
        wait
    fi

    # '!' egating set -e when patching has been applied already
    ! patch -p1 --forward <${SCRIPT_DIR}/0001-detectron2-fpn-bottom-up-separate.patch
    
    pip3 install -e .

    # back to project root
    cd ${SCRIPT_DIR}/..


    if [ -z "$(ls -A ${MODELS_WEIGHT_DIR}/detectron2)" ]; then
        echo
        echo "Downloading model weights"
        echo
    
        # FASTER R-CNN X-101 32x8d FPN
        WEIGHT_DIR="${MODELS_WEIGHT_DIR}/detectron2/COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x/139173657"
        mkdir -p ${WEIGHT_DIR}
        wget -nc https://dl.fbaipublicfiles.com/detectron2/COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x/139173657/model_final_68b088.pkl -P ${WEIGHT_DIR}

        # FASTER R-CNN R-50 FPN
        WEIGHT_DIR="${MODELS_WEIGHT_DIR}/detectron2/COCO-Detection/faster_rcnn_R_50_FPN_3x/137849458"
        mkdir -p ${WEIGHT_DIR}
        wget -nc https://dl.fbaipublicfiles.com/detectron2/COCO-Detection/faster_rcnn_R_50_FPN_3x/137849458/model_final_280758.pkl -P ${WEIGHT_DIR}

        # MASK R-CNN X-101 32x8d FPN
        WEIGHT_DIR="${MODELS_WEIGHT_DIR}/detectron2/COCO-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_3x/139653917"
        mkdir -p ${WEIGHT_DIR}
        wget -nc https://dl.fbaipublicfiles.com/detectron2/COCO-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_3x/139653917/model_final_2d9806.pkl -P ${WEIGHT_DIR}

        # MASK R-CNN R-50 FPN
        WEIGHT_DIR="${MODELS_WEIGHT_DIR}/detectron2/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x/137849600"
        mkdir -p ${WEIGHT_DIR}
        wget -nc https://dl.fbaipublicfiles.com/detectron2/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x/137849600/model_final_f10217.pkl -P ${WEIGHT_DIR}

    else
        echo
        echo "Detectron2 Weights directory not empty, using existing models"
        echo
    fi
fi

if [ ${MODEL} == "JDE" ] || [ ${MODEL} == "all" ]; then

    echo
    echo "Installing JDE"
    echo


    # install dependent packages
    pip3 install numpy motmetrics numba lap opencv-python munkres

    # install cython manually from source code with patch
    if [ -z "$(ls -A ${SCRIPT_DIR}/cython_bbox)" ]; then
        git clone https://github.com/samson-wang/cython_bbox.git ${SCRIPT_DIR}/cython_bbox
    fi

    cd ${SCRIPT_DIR}/cython_bbox
    # cython-bbox 0.1.3
    git checkout 9badb346a9222c98f828ba45c63fe3b7f2790ea2

    # '!' negating set -e when patching has been applied already
    ! patch -p1 --forward <../0001-compatible-with-numpy-1.24.1.patch
    pip3 install -e .

    # clone
    if [ -z "$(ls -A ${MODELS_SOURCE_DIR}/Towards-Realtime-MOT)" ]; then
        git clone https://github.com/Zhongdao/Towards-Realtime-MOT.git ${MODELS_SOURCE_DIR}/Towards-Realtime-MOT
    fi
    cd ${MODELS_SOURCE_DIR}/Towards-Realtime-MOT

    # git checkout
    echo
    echo "checkout branch compatible with MPEG FCM"
    echo
    git -c advice.detachedHead=false checkout c2654cdd7b69d39af669cff90758c04436025fe1

    # Apply patch to interface with compressai-vision

    # '!' negating set -e when patching has been applied already
    ! patch -p1 --forward <${SCRIPT_DIR}/0001-interface-with-compressai-vision.patch

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
    # NOTE commmented out for now as downloading via wget is blocked by provider
    # if [ -z "$(ls -A ${MODELS_WEIGHT_DIR}/jde)"]; then

    #     echo
    #     echo "Downloading weights..."
    #     echo

    #     WEIGHT_DIR="${MODELS_WEIGHT_DIR}/jde"
    #     mkdir -p ${WEIGHT_DIR}

    #     FILEID='1nlnuYfGNuHWZztQHXwVZSL_FvfE551pA'
    #     OUTFILE='jde.1088x608.uncertainty.pt'
    #     wget -nc --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id='${FILEID} -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=${FILEID}" -O ${WEIGHT_DIR}/${OUTFILE} && rm -rf /tmp/cookies.txt
    # else
    #     echo
    #     echo "JDE Weights directory not empty, using existing model"
    #     echo
    # fi
fi


echo
echo "Installing compressai-vision"
echo

pip3 install -e "${SCRIPT_DIR}/.."
pip3 install ptflops
echo
echo "NOTE: JDE pretrained weights can't be downloaded automatically"
echo "The file can be downloaded from the following link:"
echo "https://docs.google.com/uc?export=download&id=1nlnuYfGNuHWZztQHXwVZSL_FvfE551pA"
echo "and placed in the corresponding directory: ${MODELS_WEIGHT_DIR}/jde/jde.1088x608.uncertainty.pt"
echo
