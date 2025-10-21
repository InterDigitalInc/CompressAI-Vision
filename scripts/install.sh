#!/usr/bin/env bash
#
# This clones and build model architectures and gets pretrained weights
set -eu

SCRIPT_PATH="${BASH_SOURCE[0]:-${0}}"
SCRIPT_DIR=$(cd -- "$(dirname -- "${SCRIPT_PATH}")" &> /dev/null && pwd)

# --- Configuration ---
# Central array for all vision models
VISION_MODELS=(detectron2 jde yolox mmpose segment_anything)

# Default versions
TORCH_VERSION="2.0.0"
TORCHVISION_VERSION="0.15.1"
CUDA_VERSION=""
MODEL="all"
CPU="False"
COMPRESSAI_VISION_ROOT_DIR=$(cd "${SCRIPT_DIR}/.." && pwd)
MODELS_PARENT_DIR="${COMPRESSAI_VISION_ROOT_DIR}"
NO_PREPARE="False"
NO_INSTALL="False"
DOWNLOAD_WEIGHTS="True"
FCM_CTTC="False" # Install all models in conformance with MPEG FCM Common Test and Training Conditions

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
                [-m|--model, vision models to install, (detectron2/jde/yolox/mmpose/segment-anything/all) default=all]
                [-t|--torch torch version, default="2.0.0"]
                [--torchvision torchvision version, default="0.15.1"]
                [--cpu) build for cpu only)]
                [--cuda_version) provide cuda version e.g. "11.8", default: check nvcc output)]
                [--detectron2_url use this if you want to specify a pre-built detectron2 (find at
                    "https://detectron2.readthedocs.io/en/latest/tutorials/install.html#install-pre-built-detectron2-linux-only"),
                    not required for regular versions derived from cuda and torch versions above.
                    default:"https://dl.fbaipublicfiles.com/detectron2/wheels/cu102/torch1.9/index.html"]
                [--models_dir directory to install vision models to, default: compressai_vision_root]
                [--no-install) do not install (i.e. useful for only preparing source code by downloading and patching
                [--no-weights) prevents the installation script from downloading vision model parameters]
                [--fcm-cttc) Install all models in conformance with MPEG FCM Common Test and Training Conditions:
                             Torch 2.0.0, Torchvision 0.15.1, (CUDA 11.8 or CPU)]


EXAMPLE         [bash install.sh -m detectron2 -t "1.9.1" --cuda_version "11.8" --compressai /path/to/compressai]
FCM EXAMPLE     [bash install.sh --fcm-cttc (--cpu)]

_EOF_
            exit;
            ;;
        -m|--model) shift; MODEL="${1//,/ }"; shift; ;;
        -t|--torch) shift; TORCH_VERSION="$1"; shift; ;;
        --torchvision) shift; TORCHVISION_VERSION="$1"; shift; ;;
        --cpu) CPU="True"; shift; ;;
        --cuda_version) shift; CUDA_VERSION="$1"; shift; ;;
        --models_dir) shift; MODELS_PARENT_DIR="$1"; shift; ;;
        --no-prepare) NO_PREPARE="True"; shift; ;;
        --no-install) NO_INSTALL="True"; shift; ;;
        --no-weights) DOWNLOAD_WEIGHTS="False"; shift; ;;
        --fcm-cttc) FCM_CTTC="True"; shift; ;;
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
a7bf3b02f3ebf1267aba913ff637d9a2d5c33d3173bb679e46d9f338c26f262e  segment_anything/sam_vit_h_4b8939.pth                                                               https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
"


MODELS_SOURCE_DIR="${MODELS_PARENT_DIR}/models"
MODELS_WEIGHT_DIR="${MODELS_PARENT_DIR}/weights"

# pip3 is the default package manager, run install_uv.sh for uv
PACKAGE_MANAGER="${PACKAGE_MANAGER:-pip3}"

detect_env() {
    if [[ -n "${ENV_DETECTED:-}" ]]; then
        return
    fi

    if [[ "${PACKAGE_MANAGER}" == "pip3" ]]; then
        PIP=(pip3)
        MIM=(mim)
    elif [[ "${PACKAGE_MANAGER}" == "uv" ]]; then
        PIP=(uv pip)
        MIM=(uv run --no-sync mim)
    else
        echo "[ERROR] Unknown package manager: ${PACKAGE_MANAGER}. Please use 'pip3' or 'uv'."
        exit 1
    fi

    if [ "${CPU}" == "True" ]; then
        BUILD_SUFFIX="cpu"
    else
        detect_cuda_version
        BUILD_SUFFIX="cu${CUDA_VERSION//./}"
    fi

    ENV_DETECTED="True"
}

main () {

    detect_env

    if [[ "${FCM_CTTC}" == "True" ]]; then
        configure_fcm_cttc
    fi

    if [[ "${NO_PREPARE}" == "False" ]]; then
        run_prepare
    else
        echo "Skipping preparation due to --no-prepare flag."
    fi

    if [[ "${NO_INSTALL}" == "False" ]]; then
        run_install
    else
        echo "Skipping installation due to --no-install flag."
    fi

    if [ "${DOWNLOAD_WEIGHTS}" == "True" ]; then
        download_weights
    fi
}

configure_fcm_cttc() {
    echo "FCM CTTC Mode Enabled: Enforcing strict versions for all models."
    TORCH_VERSION="2.0.0"
    # Correct torchvision version for torch 2.0.0
    TORCHVISION_VERSION="0.15.1"
    MODEL="detectron2 jde yolox"
    
    if [ ${CPU} == "False" ]; then
        CTTC_CUDA_VERSION="11.8"

        # Verify CUDA version
        if [[ "${CUDA_VERSION}" != "${CTTC_CUDA_VERSION}" ]]; then
            echo "[ERROR] FCM CTTC Mode requires CUDA ${CTTC_CUDA_VERSION}, but detected ${CUDA_VERSION}."
            echo "Please ensure that your environment has CUDA ${CTTC_CUDA_VERSION} installed."
            exit 1
        fi
    fi
}

run_prepare() {
    detect_env
    mkdir -p "${MODELS_SOURCE_DIR}"
    
    # for now, prepare all models to avoid issues with uv sync
    # echo "Selected Models to be prepared: ${MODEL}"

    for model in "${VISION_MODELS[@]}"; do
        # if [[ " ${MODEL,,} " == *" ${model} "* ]] || [[ "${MODEL,,}" == "all" ]]; then
            # JDE has a dependency on cython_bbox, so prepare it first.
            if [[ "${model}" == "jde" ]]; then
                prepare_cython_bbox
            fi
            "prepare_${model}"
        # fi
    done
}

run_install () {
    detect_env
    "${PIP[@]}" install -U pip wheel setuptools
    
    echo "Selected Models to be installed: ${MODEL}"

    if [[ "${PACKAGE_MANAGER}" == "pip3" ]]; then
        install_torch
    elif [[ "${PACKAGE_MANAGER}" == "uv" ]]; then
        uv sync --extra="${BUILD_SUFFIX}"
    fi

    for model in "${VISION_MODELS[@]}"; do
        if [[ " ${MODEL,,} " == *" ${model} "* ]] || [[ "${MODEL,,}" == "all" ]]; then
            "install_${model}"
        fi
    done
    
    echo
    echo "Installing compressai"
    echo
    if [[ "${PACKAGE_MANAGER}" == "pip3" ]]; then
        "${PIP[@]}" install -e "${COMPRESSAI_VISION_ROOT_DIR}/compressai"
    elif [[ "${PACKAGE_MANAGER}" == "uv" ]]; then
        echo "Building compressai C++ extensions from source..."
        "${PIP[@]}" install "pybind11>=2.6.0" "setuptools>=68" wheel setuptools
        cd "${COMPRESSAI_VISION_ROOT_DIR}/compressai"
        rm -rf build/ **/*.so
        "${PIP[@]}" install -e . --no-build-isolation
        cd "${COMPRESSAI_VISION_ROOT_DIR}"
    fi
    "${PIP[@]}" list | grep "^compressai "

    echo
    echo "Installing compressai-vision"
    echo
    if [[ "${PACKAGE_MANAGER}" == "pip3" ]]; then
        "${PIP[@]}" install -e "${COMPRESSAI_VISION_ROOT_DIR}" --no-build-isolation
    elif [[ "${PACKAGE_MANAGER}" == "uv" ]]; then
        echo "Already installed by initial uv sync."
    fi
    "${PIP[@]}" list | grep "^compressai-vision "

    if [[ "${PACKAGE_MANAGER}" == "uv" ]]; then
        echo
        echo "Detect differences from uv.lock:"
        echo "uv sync --inexact --extra=${BUILD_SUFFIX} --dry-run"
        uv sync --inexact --extra="${BUILD_SUFFIX}" --dry-run
        uv sync --inexact --extra="${BUILD_SUFFIX}"
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
    "${PIP[@]}" install "torch==${TORCH_VERSION}" "torchvision==${TORCHVISION_VERSION}" --index-url "https://download.pytorch.org/whl/${BUILD_SUFFIX}"

}

prepare_detectron2 () {
    echo
    echo "Preparing detectron2 for installation"
    echo


    if [ -n "$(ls -A "${MODELS_SOURCE_DIR}/detectron2")" ]; then
        echo "Source directory already exists: ${MODELS_SOURCE_DIR}/detectron2"
        return
    fi
    
    git clone --single-branch --branch main https://github.com/facebookresearch/detectron2.git "${MODELS_SOURCE_DIR}/detectron2"
    cd "${MODELS_SOURCE_DIR}/detectron2"
    if [[ "${FCM_CTTC}" == "True" ]]; then
        git -c advice.detachedHead=false  checkout 175b2453c2bc4227b8039118c01494ee75b08136
        git apply "${SCRIPT_DIR}/install_utils/patches/0001-detectron2-fpn-bottom-up-separate.patch" || echo "Patch could not be applied. Possibly already applied."
    fi
    cd "${COMPRESSAI_VISION_ROOT_DIR}"
}

install_detectron2 () {
    echo
    echo "Installing detectron2"
    echo

    cd "${MODELS_SOURCE_DIR}/detectron2"
    cp ${SCRIPT_DIR}/install_utils/detectron2_pyproject.toml ./pyproject.toml

    if [[ "${PACKAGE_MANAGER}" == "pip3" ]]; then
        "${PIP[@]}" install --no-build-isolation -e .
    elif [[ "${PACKAGE_MANAGER}" == "uv" ]]; then
        cd "${COMPRESSAI_VISION_ROOT_DIR}"
        uv sync --inexact --group=models-detectron2
    fi

    cd "${COMPRESSAI_VISION_ROOT_DIR}"
}

prepare_cython_bbox () {
    echo
    echo "Preparing cython_bbox for installation"
    echo

    if [ -d "${SCRIPT_DIR}/cython_bbox" ] && [ -n "$(ls -A "${SCRIPT_DIR}/cython_bbox")" ]; then
        echo "Source directory already exists: ${SCRIPT_DIR}/cython_bbox"
        return
    fi

    git clone https://github.com/samson-wang/cython_bbox.git "${SCRIPT_DIR}/cython_bbox"
    cd "${SCRIPT_DIR}/cython_bbox"
    # cython-bbox 0.1.3
    git checkout 9badb346a9222c98f828ba45c63fe3b7f2790ea2
    git apply "${SCRIPT_DIR}/install_utils/patches/0001-cython_bbox-compatible-with-numpy-1.24.1.patch" || echo "Patch could not be applied. Possibly already applied."
    cd "${COMPRESSAI_VISION_ROOT_DIR}"
}

install_cython_bbox() {
    echo
    echo "Installing cython_bbox (required by JDE)"
    echo

    cd "${SCRIPT_DIR}/cython_bbox"

    if [[ "${PACKAGE_MANAGER}" == "pip3" ]]; then
        "${PIP[@]}" install cython numpy
        "${PIP[@]}" install -e .
    elif [[ "${PACKAGE_MANAGER}" == "uv" ]]; then
        echo "cython-bbox is installed later during JDE installation."
    fi

    cd "${COMPRESSAI_VISION_ROOT_DIR}"
}

prepare_jde () {
    echo
    echo "Preparing JDE for installation"
    echo

    if [ -d "${MODELS_SOURCE_DIR}/Towards-Realtime-MOT" ] && [ -n "$(ls -A "${MODELS_SOURCE_DIR}/Towards-Realtime-MOT")" ]; then
        echo "Source directory already exists: ${MODELS_SOURCE_DIR}/Towards-Realtime-MOT"
        return
    fi

    git clone https://github.com/Zhongdao/Towards-Realtime-MOT.git "${MODELS_SOURCE_DIR}/Towards-Realtime-MOT"
    cd "${MODELS_SOURCE_DIR}/Towards-Realtime-MOT"
    git -c advice.detachedHead=false checkout c2654cdd7b69d39af669cff90758c04436025fe1
    git apply "${SCRIPT_DIR}/install_utils/patches/0000-jde-package.patch" || echo "Patch could not be applied. Possibly already applied."
    git apply "${SCRIPT_DIR}/install_utils/patches/0001-jde-interface-with-compressai-vision.patch" || echo "Patch could not be applied. Possibly already applied."
    cd "${COMPRESSAI_VISION_ROOT_DIR}"
}

install_jde () {
    install_cython_bbox

    echo
    echo "Installing JDE"
    echo

    cd "${MODELS_SOURCE_DIR}/Towards-Realtime-MOT"

    if [[ "${PACKAGE_MANAGER}" == "pip3" ]]; then
        "${PIP[@]}" install numpy motmetrics numba lap opencv-python munkres
        "${PIP[@]}" install --no-build-isolation -e .
    elif [[ "${PACKAGE_MANAGER}" == "uv" ]]; then
        cd "${COMPRESSAI_VISION_ROOT_DIR}"
        uv sync --inexact --group=models-jde
    fi

    cd "${COMPRESSAI_VISION_ROOT_DIR}"
}

prepare_yolox () {
    echo
    echo "Preparing YOLOX for installation"
    echo

    if [ -d "${MODELS_SOURCE_DIR}/yolox" ] && [ -n "$(ls -A "${MODELS_SOURCE_DIR}/yolox")" ]; then
        echo "Source directory already exists: ${MODELS_SOURCE_DIR}/yolox"
        return
    fi

    git clone https://github.com/Megvii-BaseDetection/yolox.git "${MODELS_SOURCE_DIR}/yolox"
    cd "${MODELS_SOURCE_DIR}/yolox"
    git reset --hard d872c71bf63e1906ef7b7bb5a9d7a529c7a59e6a
    cd "${COMPRESSAI_VISION_ROOT_DIR}"
}

install_yolox () {
    echo
    echo "Installing YOLOX (reference: https://github.com/Megvii-BaseDetection/YOLOX)"
    echo

    cd "${MODELS_SOURCE_DIR}/yolox"

    if [[ "${PACKAGE_MANAGER}" == "pip3" ]]; then
        # miminum requirments - no onnx, etc.
        cp "${SCRIPT_DIR}/install_utils/yolox_requirements.txt" requirements.txt
        cp ${SCRIPT_DIR}/install_utils/yolox_pyproject.toml ./pyproject.toml
        "${PIP[@]}" install --no-build-isolation -e .
    elif [[ "${PACKAGE_MANAGER}" == "uv" ]]; then
        cd "${COMPRESSAI_VISION_ROOT_DIR}"
        uv sync --inexact --group=models-yolox
    fi

    cd "${COMPRESSAI_VISION_ROOT_DIR}"
}

prepare_mmpose () {
    echo
    echo "Preparing MMPOSE for installation"
    echo

    if [ -d "${MODELS_SOURCE_DIR}/mmpose" ] && [ -n "$(ls -A "${MODELS_SOURCE_DIR}/mmpose")" ]; then
        echo "Source directory already exists: ${MODELS_SOURCE_DIR}/mmpose"
        return
    fi

    git clone https://github.com/open-mmlab/mmpose.git "${MODELS_SOURCE_DIR}/mmpose"
    cd "${MODELS_SOURCE_DIR}/mmpose"
    git reset --hard 71ec36ebd63c475ab589afc817868e749a61491f
    cd "${COMPRESSAI_VISION_ROOT_DIR}"
}

install_mmpose () {
    echo
    echo "Installing MMPOSE (reference: https://github.com/open-mmlab/mmpose/tree/main)"
    echo

    cd "${MODELS_SOURCE_DIR}/mmpose"

    if [[ "${PACKAGE_MANAGER}" == "pip3" ]]; then
        "${PIP[@]}" install -U openmim
        # miminum requirments - no onnx, etc.
        "${PIP[@]}" install -r requirements.txt
        "${PIP[@]}" install -v -e .
        # during the installation isort version might be overwritten.
        # hence make sure back to the isort=5.13.2
        "${PIP[@]}" install isort==5.13.2
    elif [[ "${PACKAGE_MANAGER}" == "uv" ]]; then
        cd "${COMPRESSAI_VISION_ROOT_DIR}"
        uv sync --inexact --group=models-mmpose
    fi

    "${MIM[@]}" install "mmcv==2.0.1"
    "${MIM[@]}" install "mmdet==3.1.0"

    cd "${COMPRESSAI_VISION_ROOT_DIR}"
}

prepare_segment_anything () {
    echo
    echo "Preparing Segment Anything for installation"
    echo

    if [ -d "${MODELS_SOURCE_DIR}/segment_anything" ] && [ -n "$(ls -A "${MODELS_SOURCE_DIR}/segment_anything")" ]; then
        echo "Source directory already exists: ${MODELS_SOURCE_DIR}/segment_anything"
        return
    fi

    git clone https://github.com/facebookresearch/segment-anything.git "${MODELS_SOURCE_DIR}/segment_anything"
    cd "${MODELS_SOURCE_DIR}/segment_anything"
    git reset --hard dca509fe793f601edb92606367a655c15ac00fdf
    cd "${COMPRESSAI_VISION_ROOT_DIR}"
}

install_segment_anything () {
    echo
    echo "Installing Segment Anything (reference: https://github.com/facebookresearch/segment-anything/commits/main/)"
    echo

    cd "${MODELS_SOURCE_DIR}/segment_anything"

    if [[ "${PACKAGE_MANAGER}" == "pip3" ]]; then
        "${PIP[@]}" install -e .
    elif [[ "${PACKAGE_MANAGER}" == "uv" ]]; then
        cd "${COMPRESSAI_VISION_ROOT_DIR}"
        uv sync --inexact --group=models-segment-anything
    fi

    cd "${COMPRESSAI_VISION_ROOT_DIR}"
}


download_weights () {
    detect_env
    mkdir -p "${MODELS_WEIGHT_DIR}"
    cd "${MODELS_WEIGHT_DIR}/"
    
    for model in detectron2 jde mmpose yolox segment_anything; do
        if ! [[ ",${MODEL,,}," == *",${model},"* ]] && [[ ",${MODEL,,}," != *",all,"* ]]; then
            continue
        fi

        echo
        echo
        echo
        echo "Downloading model weights for ${model}..."
        echo

        FILTER="[0-9a-fA-F]* ${model}/"
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

    cd "${COMPRESSAI_VISION_ROOT_DIR}"
}


if [[ "${__SOURCE_ONLY__:-0}" -eq 0 ]]; then
    main "$@"
fi