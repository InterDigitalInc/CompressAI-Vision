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
DOWNLOAD_WEIGHTS="True"

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
                [-m|--model, vision models to install, (detectron2/jde/yolox/mmpose/all) default=all]
                [-t|--torch torch version, default="2.0.0"]
                [--torchvision torchvision version, default="0.15.1"]
                [--cpu) build for cpu only)]
                [--cuda) provide cuda version e.g. "11.8", default: check nvcc output)]
                [--detectron2_url use this if you want to specify a pre-built detectron2 (find at
                    "https://detectron2.readthedocs.io/en/latest/tutorials/install.html#install-pre-built-detectron2-linux-only"),
                    not required for regular versions derived from cuda and torch versions above.
                    default:"https://dl.fbaipublicfiles.com/detectron2/wheels/cu102/torch1.9/index.html"]
                [--models_dir directory to install vision models to, default: compressai_vision_root]
                [--no-weights) prevents the installation script from downloading vision model parameters]


EXAMPLE         [bash install_models.sh -m detectron2 -t "1.9.1" --cuda "11.8" --compressai /path/to/compressai]

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
        --no-weights) DOWNLOAD_WEIGHTS="False"; shift; ;;
        *) echo "[ERROR] Unknown parameter $1"; exit; ;;
    esac;
done;


WEIGHTS="
3c25caca37baabbff3e22cc9eb0923db165a0c18b867871a3bf3570bac9b7ef0  detectron2/COCO-Detection/faster_rcnn_R_50_FPN_3x/137849458/model_final_280758.pkl                  https://dl.fbaipublicfiles.com/detectron2/COCO-Detection/faster_rcnn_R_50_FPN_3x/137849458/model_final_280758.pkl
fe5ad56ff746aa55c5f453b01f8395134e9281d240dbeb473411d4a6b262c9dc  detectron2/COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x/139173657/model_final_68b088.pkl           https://dl.fbaipublicfiles.com/detectron2/COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x/139173657/model_final_68b088.pkl
9a737e290372f1f70994ebcbd89d8004dbb3ae30a605fd915a190fa4a782dd66  detectron2/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x/137849600/model_final_f10217.pkl         https://dl.fbaipublicfiles.com/detectron2/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x/137849600/model_final_f10217.pkl
12f6e1811baf1b4d329c3f5ac5ec52d8f634d3cedc82a13fff55d0c05d84f442  detectron2/COCO-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_3x/139653917/model_final_2d9806.pkl  https://dl.fbaipublicfiles.com/detectron2/COCO-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_3x/139653917/model_final_2d9806.pkl
808c675e647298688589c895c9581f7f3963995c5708bc53f66449200321d147  detectron2/COCO-PanopticSegmentation/panoptic_fpn_R_101_3x/139514519/model_final_cafdb1.pkl         https://dl.fbaipublicfiles.com/detectron2/COCO-PanopticSegmentation/panoptic_fpn_R_101_3x/139514519/model_final_cafdb1.pkl
6b135b0affa38899b607010c86c2f8dbc1c06956bad9ca1edd45b01e626933f1  jde/jde.1088x608.uncertainty.pt                                                                     https://drive.usercontent.google.com/download?export=download&confirm=t&id=1nlnuYfGNuHWZztQHXwVZSL_FvfE551pA
516a421f8717548300c3ee6356a3444ac539083d4a9912f8ca1619ee63d0986d  mmpose/rtmo_coco/rtmo-l_16xb16-600e_coco-640x640-516a421f_20231211.pth                              https://download.openmmlab.com/mmpose/v1/projects/rtmo/rtmo-l_16xb16-600e_coco-640x640-516a421f_20231211.pth
b5905e9faf500a2608c93991f91a41a6150bcd2dd30986865a73becd94542fa1  yolox/darknet53/yolox_darknet.pth                                                                   https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_darknet.pth
"


MODELS_SOURCE_DIR=${MODELS_ROOT_DIR}/models
MODELS_WEIGHT_DIR=${MODELS_ROOT_DIR}/weights

mkdir -p ${MODELS_SOURCE_DIR}
mkdir -p ${MODELS_WEIGHT_DIR}


main () {
    uv sync

    uv pip install -U pip wheel

    install_torch

    for model in detectron2 jde yolox mmpose; do
        if [[ ",${MODEL,,}," == *",${model},"* ]] || [[ ",${MODEL,,}," == *",all,"* ]]; then
            "install_${model}"
        fi
    done

    echo
    echo "Installing compressai"
    echo
    uv pip install -e "${SCRIPT_DIR}/../compressai"

    echo
    echo "Installing compressai-vision"
    echo

    uv pip install -e "${SCRIPT_DIR}/.."

    echo
    echo "uv sync --inexact --dry-run (check for differences from uv.lock)"
    echo
    uv sync --inexact --dry-run

    if [ "${DOWNLOAD_WEIGHTS}" == "True" ]; then
        download_weights
    fi
}


detect_cuda_version () {
    if [ -n "${CUDA_VERSION}" ]; then
        echo "Using specified CUDA version: ${CUDA_VERSION}"
        return
    fi
    echo "Detecting CUDA version..."
    if [ -z "$(command -v nvcc)" ]; then
        echo "nvcc not found. Please ensure CUDA is installed, specify the CUDA version using --cuda, or source scripts/env_cuda.sh."
        exit 1
    fi
    CUDA_VERSION=$(nvcc --version | sed -n 's/^.*release \([0-9]\+\.[0-9]\+\).*$/\1/p')
    if [ -z "${CUDA_VERSION}" ]; then
        echo "Could not detect CUDA version. Please specify the CUDA version using --cuda or source scripts/env_cuda.sh."
        exit 1
    fi
    echo "Detected CUDA version: ${CUDA_VERSION}"
}

install_torch () {
    if [ "${CPU}" == "True" ]; then
        uv pip install "torch==${TORCH_VERSION}" "torchvision==${TORCHVISION_VERSION}" --index-url https://download.pytorch.org/whl/cpu
    else
        detect_cuda_version
        uv pip install torch==${TORCH_VERSION}+cu${CUDA_VERSION//./} torchvision==${TORCHVISION_VERSION}+cu${CUDA_VERSION//./} --extra-index-url https://download.pytorch.org/whl/cu${CUDA_VERSION//./}
    fi
}
    
install_detectron2 () {
    echo
    echo "Installing detectron2"
    echo

    # clone
    if [ -z "$(ls -A ${MODELS_SOURCE_DIR}/detectron2)" ]; then
        git clone --single-branch --branch main https://github.com/facebookresearch/detectron2.git ${MODELS_SOURCE_DIR}/detectron2
    fi
    cd ${MODELS_SOURCE_DIR}/detectron2

    echo
    echo "checkout the version used for MPEG FCM"
    echo
    git -c advice.detachedHead=false  checkout 175b2453c2bc4227b8039118c01494ee75b08136

    git apply "${SCRIPT_DIR}/patches/0001-detectron2-fpn-bottom-up-separate.patch" || echo "Patch could not be applied. Possibly already applied."

    uv pip install --no-build-isolation .

    # back to project root
    cd ${SCRIPT_DIR}/..
}

install_cython_bbox() {
    echo
    echo "Installing cython_bbox (required by JDE)"
    echo

    # install cython manually from source code with patch
    if [ -z "$(ls -A ${SCRIPT_DIR}/cython_bbox)" ]; then
        git clone https://github.com/samson-wang/cython_bbox.git ${SCRIPT_DIR}/cython_bbox
    fi

    cd ${SCRIPT_DIR}/cython_bbox
    # cython-bbox 0.1.3
    git checkout 9badb346a9222c98f828ba45c63fe3b7f2790ea2

    git apply "${SCRIPT_DIR}/patches/0001-cython_bbox-compatible-with-numpy-1.24.1.patch" || echo "Patch could not be applied. Possibly already applied."
    
    uv pip install --no-build-isolation .

    cd "${SCRIPT_DIR}/.."
}

install_jde () {
    echo
    echo "Installing JDE"
    echo

    install_cython_bbox
    
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

    # Convert to installable python package.
    git apply "${SCRIPT_DIR}/patches/0000-jde-package.patch" || echo "Patch could not be applied. Possibly already applied."

    # Apply patch to interface with compressai-vision
    git apply "${SCRIPT_DIR}/patches/0001-jde-interface-with-compressai-vision.patch" || echo "Patch could not be applied. Possibly already applied."
    
    uv pip install --no-build-isolation .

    # back to project root
    cd ${SCRIPT_DIR}/..
}

install_yolox () {
    echo
    echo "Installing YOLOX (reference: https://github.com/Megvii-BaseDetection/YOLOX)"
    echo

    uv sync --inexact --group=models-yolox

    # clone
    if [ -z "$(ls -A ${MODELS_SOURCE_DIR}/yolox)" ]; then
        git clone https://github.com/Megvii-BaseDetection/yolox.git ${MODELS_SOURCE_DIR}/yolox
        
        # checkout specific commit on Nov.19, 2024 for now to avoid compatibility in the future.
        cd ${MODELS_SOURCE_DIR}/yolox
        git reset --hard d872c71bf63e1906ef7b7bb5a9d7a529c7a59e6a
        cd ${SCRIPT_DIR}/..
    fi

    cd ${MODELS_SOURCE_DIR}/yolox

    uv pip install --no-build-isolation .

    # back to project root
    cd ${SCRIPT_DIR}/..
}

install_mmpose () {
    echo
    echo "Installing MMPOSE (reference: https://github.com/open-mmlab/mmpose/tree/main)"
    echo

    # clone
    if [ -z "$(ls -A ${MODELS_SOURCE_DIR}/mmpose)" ]; then
        git clone https://github.com/open-mmlab/mmpose.git ${MODELS_SOURCE_DIR}/mmpose
        
        # checkout specific commit version to avoid compatibility in the future.
        cd ${MODELS_SOURCE_DIR}/mmpose
        git reset --hard 71ec36ebd63c475ab589afc817868e749a61491f
        cd ${SCRIPT_DIR}/..
    fi

    cd ${MODELS_SOURCE_DIR}/mmpose

    uv sync --inexact --group=models-mmpose
    uv pip install --no-build-isolation .
    uv run --no-sync mim install "mmcv==2.0.1"
    uv run --no-sync mim install mmdet==3.1.0

    # back to project root
    cd ${SCRIPT_DIR}/..
}

download_weights () {
    cd "${MODELS_WEIGHT_DIR}/"

    for model in detectron2 jde mmpose yolox; do
        if ! [[ ",${MODEL,,}," == *",${model},"* ]] && [[ ",${MODEL,,}," != *",all,"* ]]; then
            continue
        fi

        echo
        echo
        echo
        echo "Downloading model weights for ${model}..."
        echo

        FILTER="^[0-9a-fA-F]*  ${model}/"
        FILTERED_WEIGHTS=$(echo "$WEIGHTS" | grep "${FILTER}")

        echo "${FILTERED_WEIGHTS}" | while read -r entry; do
            read -r _SHA256SUM OUTPATH URL <<< "$entry"
            mkdir -p "${OUTPATH%/*}"
            if [[ -f "${OUTPATH}" ]]; then
                echo "${OUTPATH} already exists. Skipping download."
            else
                wget "${URL}" -O "${OUTPATH}" || {
                    echo "Failed to download ${OUTPATH} from ${URL}"
                    echo "Continuing other downloads..."
                }
            fi
        done

        echo
        echo "Verifying checksums for ${model}..."
        echo

        if ! echo "${FILTERED_WEIGHTS}" | awk '{print $1 "  " $2}' | sha256sum --check; then
            echo
            echo "Checksum verification failed for ${model}."
            echo "Consider downloading the weights manually inside the directory ${MODELS_WEIGHT_DIR}/:"
            echo "${FILTERED_WEIGHTS}"
            exit 1
        fi
    done

    cd "${SCRIPT_DIR}/.."
}

