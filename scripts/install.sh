#!/usr/bin/env bash
#
# This clones and build model architectures and gets pretrained weights
set -eu

SCRIPT_PATH="${BASH_SOURCE[0]:-${0}}"
SCRIPT_DIR=$(cd -- "$(dirname -- "${SCRIPT_PATH}")" &> /dev/null && pwd)

# --- Configuration ---
# Central array for all vision models
VISION_MODELS=(
    detectron2
    jde
    yolox
    mmpose
    segment_anything
    sam2
    fasterrcnn_mobilenet_v3_large_320_fpn
    lraspp_mobilenet_v3_large
    efficientvit_sam_l0
    efficientvit_sam_l1
    efficientvit_sam_l2
    efficientvit_sam_xl0
    efficientvit_sam_xl1
)

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
                [-m|--model, vision models to install, (detectron2/jde/yolox/mmpose/segment_anything/sam2/fasterrcnn_mobilenet_v3_large_320_fpn/lraspp_mobilenet_v3_large/efficientvit_sam_l0/efficientvit_sam_l1/efficientvit_sam_l2/efficientvit_sam_xl0/efficientvit_sam_xl1/all) default=all]
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

MODEL="${MODEL//segment-anything/segment_anything}"
MODEL="${MODEL//faster-rcnn-mobilenet-v3-large-320-fpn/fasterrcnn_mobilenet_v3_large_320_fpn}"
MODEL="${MODEL//lr-aspp-mobilenet-v3-large/lraspp_mobilenet_v3_large}"
MODEL="${MODEL//efficientvit-sam-/efficientvit_sam_}"

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
a2345aede8715ab1d5d31b4a509fb160c5a4af1970f199d9054ccfb746c004c5  sam2/sam2.1_hiera_base_plus.pt                                                                      https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_base_plus.pt
2647878d5dfa5098f2f8649825738a9345572bae2d4350a2468587ece47dd318  sam2/sam2.1_hiera_large.pt                                                                          https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt
6d1aa6f30de5c92224f8172114de081d104bbd23dd9dc5c58996f0cad5dc4d38  sam2/sam2.1_hiera_small.pt                                                                          https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_small.pt
7402e0d864fa82708a20fbd15bc84245c2f26dff0eb43a4b5b93452deb34be69  sam2/sam2.1_hiera_tiny.pt                                                                           https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_tiny.pt
ba17f9fb28832baf5573e8a4212cad7740d606dd95b9c6d13c5465f603627813  efficientvit/source_json_file/coco_vitdet.json                                                     https://huggingface.co/mit-han-lab/efficientvit-sam/resolve/main/source_json_file/coco_vitdet.json?download=true
c4f994b01a16d48bcf2fbbb089448cfbf58fae5811edfa8113c953b8b8cc64b8  efficientvit/sam/l0.pt                                                                              https://huggingface.co/mit-han-lab/efficientvit-sam/resolve/main/efficientvit_sam_l0.pt?download=true
fa151df2b9b96896accd505470755cdd293a4c50cdabac95478f3fbd86d9152b  efficientvit/sam/l1.pt                                                                              https://huggingface.co/mit-han-lab/efficientvit-sam/resolve/main/efficientvit_sam_l1.pt?download=true
d4bfd842224cbb99de09acc3325e81ac5b7e8725c8fbdab152305e5d934bfe4a  efficientvit/sam/l2.pt                                                                              https://huggingface.co/mit-han-lab/efficientvit-sam/resolve/main/efficientvit_sam_l2.pt?download=true
9bc2f87bd23d6e6d1f7b5beff910f98835c7c333fac73e695bac0bec685968f8  efficientvit/sam/xl0.pt                                                                             https://huggingface.co/mit-han-lab/efficientvit-sam/resolve/main/efficientvit_sam_xl0.pt?download=true
10f42277679427c1183dcff19758e536aa86bcc4304e4b387ff48865784be05c  efficientvit/sam/xl1.pt                                                                             https://huggingface.co/mit-han-lab/efficientvit-sam/resolve/main/efficientvit_sam_xl1.pt?download=true
"


MODELS_SOURCE_DIR="${MODELS_PARENT_DIR}/models"
MODELS_WEIGHT_DIR="${MODELS_PARENT_DIR}/weights"
# SAM2 is resolved in its own environment; see install_sam2.
SAM2_VENV_DIR="${SAM2_VENV_DIR:-${COMPRESSAI_VISION_ROOT_DIR}/.venv-sam2}"

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
    
    echo "Selected Models to be prepared: ${MODEL}"

    for model in "${VISION_MODELS[@]}"; do
        if [[ " ${MODEL,,} " != *" ${model} "* ]] && [[ "${MODEL,,}" != "all" ]]; then
            continue
        fi
        # JDE has a dependency on cython_bbox, so prepare it first.
        if [[ "${model}" == "jde" ]]; then
            prepare_cython_bbox
        fi
        "prepare_${model}"
    done
}

run_install () {
    detect_env
    "${PIP[@]}" install -U pip wheel "setuptools>=68,<81"
    
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
        "${PIP[@]}" install "pybind11>=2.6.0" "setuptools>=68,<81" wheel
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
    "${PIP[@]}" install "torch==${TORCH_VERSION}" "torchvision==${TORCHVISION_VERSION}" \
        --index-url "https://download.pytorch.org/whl/${BUILD_SUFFIX}" \
        --extra-index-url "https://pypi.org/simple"

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
    git apply "${SCRIPT_DIR}/install_utils/patches/0002-detectron2-lazy-torch-import.patch" || echo "Patch could not be applied. Possibly already applied."

    if [[ "${PACKAGE_MANAGER}" == "pip3" ]]; then
        "${PIP[@]}" install --no-build-isolation -e .
    elif [[ "${PACKAGE_MANAGER}" == "uv" ]]; then
        cd "${COMPRESSAI_VISION_ROOT_DIR}"
        uv sync --inexact --group=models-detectron2 --extra="${BUILD_SUFFIX}"
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
        uv sync --inexact --group=models-jde --extra="${BUILD_SUFFIX}"
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
        uv sync --inexact --group=models-yolox --extra="${BUILD_SUFFIX}"
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
        uv sync --inexact --group=models-mmpose --extra="${BUILD_SUFFIX}"
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
        uv sync --inexact --group=models-segment-anything --extra="${BUILD_SUFFIX}"
    fi

    cd "${COMPRESSAI_VISION_ROOT_DIR}"
}

prepare_sam2 () {
    echo
    echo "Preparing Segment Anything 2 for installation"
    echo

    if [ -d "${MODELS_SOURCE_DIR}/sam2" ] && [ -n "$(ls -A "${MODELS_SOURCE_DIR}/sam2")" ]; then
        echo "Source directory already exists: ${MODELS_SOURCE_DIR}/sam2"
        return
    fi

    git clone https://github.com/facebookresearch/sam2.git "${MODELS_SOURCE_DIR}/sam2"
    cd "${MODELS_SOURCE_DIR}/sam2"
    #  Dec 16, 2024
    git reset --hard 2b90b9f5ceec907a1c18123530e92e794ad901a4
    cd "${COMPRESSAI_VISION_ROOT_DIR}"
}

install_sam2 () {
    echo
    echo "Installing Segment Anything 2 (reference: https://github.com/facebookresearch/sam2)"
    echo "Requirements: python>=3.10, as well as torch>=2.5.1 and torchvision>=0.20.1."
    echo

    # SAM2 needs torch>=2.5.1 and iopath>=0.1.10. The FCM CTTC stack pins torch
    # 2.0.0 and detectron2 pins iopath<0.1.10, so SAM2 cannot share their
    # environment; see the [tool.uv] conflicts in pyproject.toml. It is
    # installed into a separate venv instead.
    local sam2_extra
    if [ "${CPU}" == "True" ]; then
        sam2_extra="sam2-cpu"
    else
        sam2_extra="cu128"
        if [[ "${CUDA_VERSION}" != 12.* ]]; then
            echo "[WARNING] SAM2 requires torch>=2.5.1, which is built here against CUDA 12.8,"
            echo "          but the detected CUDA version is ${CUDA_VERSION}."
        fi
    fi

    if [[ "${PACKAGE_MANAGER}" == "pip3" ]]; then
        echo "[WARNING] Installing SAM2 with pip into the active venv will upgrade torch"
        echo "          past the FCM CTTC version. Use a dedicated venv for SAM2."
        cd "${MODELS_SOURCE_DIR}/sam2"
        "${PIP[@]}" install -e .
    elif [[ "${PACKAGE_MANAGER}" == "uv" ]]; then
        cd "${COMPRESSAI_VISION_ROOT_DIR}"
        UV_PROJECT_ENVIRONMENT="${SAM2_VENV_DIR}" \
            uv sync --inexact --group=models-sam2 --extra="${sam2_extra}"
        echo "SAM2 installed into ${SAM2_VENV_DIR}"
    fi

    cd "${COMPRESSAI_VISION_ROOT_DIR}"
}

prepare_torchvision_model () {
    echo
    echo "Preparing torchvision-backed models for installation"
    echo "No external source tree is required."
    echo
}

install_torchvision_model () {
    echo
    echo "Installing torchvision-backed models"
    echo "Faster R-CNN MobileNetV3-Large 320 FPN and LR-ASPP MobileNetV3-Large are provided by torchvision."
    echo

    if ! "${PIP[@]}" show torchvision >/dev/null 2>&1; then
        install_torch
    fi
}

prepare_fasterrcnn_mobilenet_v3_large_320_fpn () {
    prepare_torchvision_model
}

install_fasterrcnn_mobilenet_v3_large_320_fpn () {
    install_torchvision_model
}

prepare_lraspp_mobilenet_v3_large () {
    prepare_torchvision_model
}

install_lraspp_mobilenet_v3_large () {
    install_torchvision_model
}

prepare_efficientvit_model () {
    echo
    echo "Preparing EfficientViT for installation"
    echo

    if [ -d "${MODELS_SOURCE_DIR}/efficientvit" ] && [ -n "$(ls -A "${MODELS_SOURCE_DIR}/efficientvit")" ]; then
        echo "Source directory already exists: ${MODELS_SOURCE_DIR}/efficientvit"
        return
    fi

    git clone https://github.com/mit-han-lab/efficientvit.git "${MODELS_SOURCE_DIR}/efficientvit"
    cd "${COMPRESSAI_VISION_ROOT_DIR}"
}

install_efficientvit_model () {
    if [[ "${EFFICIENTVIT_INSTALLED:-False}" == "True" ]]; then
        echo "EfficientViT-SAM is already installed in this run. Skipping duplicate installation."
        return
    fi

    echo
    echo "Installing EfficientViT-SAM (reference: https://github.com/mit-han-lab/efficientvit)"
    echo "The upstream repository documents Python 3.10; this installer keeps dependencies minimal for Python 3.8 + CUDA 11.8."
    echo

    if [ ! -d "${MODELS_SOURCE_DIR}/segment_anything" ]; then
        prepare_segment_anything
    fi
    if ! "${PIP[@]}" show segment_anything >/dev/null 2>&1; then
        install_segment_anything
    fi

    if [ ! -d "${MODELS_SOURCE_DIR}/efficientvit" ]; then
        prepare_efficientvit_model
    fi

    "${PIP[@]}" install "timm<1.0.0" einops torchprofile scipy tqdm huggingface-hub

    cd "${MODELS_SOURCE_DIR}/efficientvit"
    "${PIP[@]}" install --no-deps -e .
    cd "${COMPRESSAI_VISION_ROOT_DIR}"

    EFFICIENTVIT_INSTALLED="True"
}

prepare_efficientvit_sam_l0 () {
    prepare_efficientvit_model
}

install_efficientvit_sam_l0 () {
    install_efficientvit_model
}

prepare_efficientvit_sam_l1 () {
    prepare_efficientvit_model
}

install_efficientvit_sam_l1 () {
    install_efficientvit_model
}

prepare_efficientvit_sam_l2 () {
    prepare_efficientvit_model
}

install_efficientvit_sam_l2 () {
    install_efficientvit_model
}

prepare_efficientvit_sam_xl0 () {
    prepare_efficientvit_model
}

install_efficientvit_sam_xl0 () {
    install_efficientvit_model
}

prepare_efficientvit_sam_xl1 () {
    prepare_efficientvit_model
}

install_efficientvit_sam_xl1 () {
    install_efficientvit_model
}

download_weights () {
    detect_env
    mkdir -p "${MODELS_WEIGHT_DIR}"
    cd "${MODELS_WEIGHT_DIR}/"

    for model in "${VISION_MODELS[@]}"; do
        if [[ " ${MODEL,,} " == *" ${model} "* ]] || [[ "${MODEL,,}" == "all" ]]; then
            echo
            echo
            echo
            echo "Downloading model weights for ${model}..."
            echo

            case "${model}" in
                fasterrcnn_mobilenet_v3_large_320_fpn|lraspp_mobilenet_v3_large)
                    FILTER="__NO_EXTERNAL_WEIGHTS__"
                    ;;
                efficientvit_sam_l0)
                    FILTER="[0-9a-fA-F]+ +(efficientvit/source_json_file/coco_vitdet.json|efficientvit/sam/l0.pt)"
                    ;;
                efficientvit_sam_l1)
                    FILTER="[0-9a-fA-F]+ +(efficientvit/source_json_file/coco_vitdet.json|efficientvit/sam/l1.pt)"
                    ;;
                efficientvit_sam_l2)
                    FILTER="[0-9a-fA-F]+ +(efficientvit/source_json_file/coco_vitdet.json|efficientvit/sam/l2.pt)"
                    ;;
                efficientvit_sam_xl0)
                    FILTER="[0-9a-fA-F]+ +(efficientvit/source_json_file/coco_vitdet.json|efficientvit/sam/xl0.pt)"
                    ;;
                efficientvit_sam_xl1)
                    FILTER="[0-9a-fA-F]+ +(efficientvit/source_json_file/coco_vitdet.json|efficientvit/sam/xl1.pt)"
                    ;;
                *)
                    FILTER="[0-9a-fA-F]* ${model}/"
                    ;;
            esac
            FILTERED_WEIGHTS=$(echo "$WEIGHTS" | grep -E "${FILTER}" || true)

            if [[ -z "${FILTERED_WEIGHTS}" ]]; then
                echo "No external weight files are managed by this script for ${model}."
                continue
            fi

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
        fi
    done

    cd "${COMPRESSAI_VISION_ROOT_DIR}"
}


if [[ "${__SOURCE_ONLY__:-0}" -eq 0 ]]; then
    main "$@"
fi
