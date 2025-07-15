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
        --no-weights) DOWNLOAD_WEIGHTS="False"; shift; ;;
        *) echo "[ERROR] Unknown parameter $1"; exit; ;;
    esac;
done;

MODELS_SOURCE_DIR=${MODELS_ROOT_DIR}/models
MODELS_WEIGHT_DIR=${MODELS_ROOT_DIR}/weights

mkdir -p ${MODELS_SOURCE_DIR}
mkdir -p ${MODELS_WEIGHT_DIR}


main () {
    pip3 install -U pip wheel

    install_torch

    if [ "${MODEL,,}" == "detectron2" ] || [ ${MODEL} == "all" ]; then
        install_detectron2
    fi

    if [ "${MODEL,,}" == "jde" ] || [ ${MODEL} == "all" ]; then
        install_jde
    fi

    if [ "${MODEL,,}" == "yolox" ] || [ ${MODEL} == "all" ]; then
        install_yolox
    fi

    if [ "${MODEL,,}" == "mmpose" ] || [ ${MODEL} == "all" ]; then
        install_mmpose
    fi

    echo
    echo "Installing compressai"
    echo
    pip3 install -e "${SCRIPT_DIR}/../compressai"

    echo
    echo "Installing compressai-vision"
    echo
    pip3 install -e "${SCRIPT_DIR}/.."
}


install_torch () {
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
    
    pip3 install -e .

    # back to project root
    cd ${SCRIPT_DIR}/..

    if [ "${DOWNLOAD_WEIGHTS}" == "True" ]; then
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

            # Panoptic FPN R-101 3x
            WEIGHT_DIR="${MODELS_WEIGHT_DIR}/detectron2/COCO-PanopticSegmentation/panoptic_fpn_R_101_3x/139514519"
            mkdir -p ${WEIGHT_DIR}
            wget -nc https://dl.fbaipublicfiles.com/detectron2/COCO-PanopticSegmentation/panoptic_fpn_R_101_3x/139514519/model_final_cafdb1.pkl -P ${WEIGHT_DIR}
        else
            echo
            echo "Detectron2 Weights directory not empty, using existing models"
            echo
        fi
    fi
}

install_jde () {
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

    git apply "${SCRIPT_DIR}/patches/0001-cython_bbox-compatible-with-numpy-1.24.1.patch" || echo "Patch could not be applied. Possibly already applied."
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
    git apply "${SCRIPT_DIR}/patches/0001-jde-interface-with-compressai-vision.patch" || echo "Patch could not be applied. Possibly already applied."

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
    if [ "${DOWNLOAD_WEIGHTS}" == "True" ]; then
    #   NOTE commmented out for now as downloading via wget is blocked by provider
    #   if [ -z "$(ls -A ${MODELS_WEIGHT_DIR}/jde)"]; then

    #     echo
    #     echo "Downloading weights..."
    #     echo

    #     WEIGHT_DIR="${MODELS_WEIGHT_DIR}/jde"
    #     mkdir -p ${WEIGHT_DIR}

    #     FILEID='1nlnuYfGNuHWZztQHXwVZSL_FvfE551pA'
    #     OUTFILE='jde.1088x608.uncertainty.pt'
    #     wget -nc --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id='${FILEID} -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=${FILEID}" -O ${WEIGHT_DIR}/${OUTFILE} && rm -rf /tmp/cookies.txt
    #   else
    #     echo
    #     echo "JDE Weights directory not empty, using existing model"
    #     echo
    #   fi
    #

        echo
        echo "NOTE: JDE pretrained weights can't be downloaded automatically"
        echo "The file can be downloaded from the following link:"
        echo "https://docs.google.com/uc?export=download&id=1nlnuYfGNuHWZztQHXwVZSL_FvfE551pA"
        echo "and placed in the corresponding directory: ${MODELS_WEIGHT_DIR}/jde/jde.1088x608.uncertainty.pt"
        echo
    fi
}

install_yolox () {
    echo
    echo "Installing YOLOX (reference: https://github.com/Megvii-BaseDetection/YOLOX)"
    echo

    # clone
    if [ -z "$(ls -A ${MODELS_SOURCE_DIR}/yolox)" ]; then
        git clone https://github.com/Megvii-BaseDetection/yolox.git ${MODELS_SOURCE_DIR}/yolox
        
        # checkout specific commit on Nov.19, 2024 for now to avoid compatibility in the future.
        cd ${MODELS_SOURCE_DIR}/yolox
        git reset --hard d872c71bf63e1906ef7b7bb5a9d7a529c7a59e6a
        cd ${SCRIPT_DIR}/..
    fi

    cd ${MODELS_SOURCE_DIR}/yolox
    # miminum requirments - no onnx, etc.
    cp ${SCRIPT_DIR}/yolox_requirements.txt requirements.txt

    pip3 install -e .
    if [ "${DOWNLOAD_WEIGHTS}" == "True" ]; then
        if [ -z "$(ls -A ${MODELS_WEIGHT_DIR}/yolox)" ]; then
            echo
            echo "Downloading model weights"
            echo

            # YOLOX-Darkent53
            WEIGHT_DIR="${MODELS_WEIGHT_DIR}/yolox/darknet53"
            mkdir -p ${WEIGHT_DIR}
            wget -nc https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_darknet.pth -P ${WEIGHT_DIR}
        else
            echo
            echo "YOLOX Weights directory is not empty, using existing models"
            echo
        fi
    fi
    # back to project root
    cd ${SCRIPT_DIR}/..
}

install_mmpose () {
    echo
    echo "Installing MMPOSE (reference: https://github.com/open-mmlab/mmpose/tree/main)"
    echo

    pip install -U openmim
    mim install "mmcv==2.0.1"

    # clone
    if [ -z "$(ls -A ${MODELS_SOURCE_DIR}/mmpose)" ]; then
        git clone https://github.com/open-mmlab/mmpose.git ${MODELS_SOURCE_DIR}/mmpose
        
        # checkout specific commit version to avoid compatibility in the future.
        cd ${MODELS_SOURCE_DIR}/mmpose
        git reset --hard 71ec36ebd63c475ab589afc817868e749a61491f
        cd ${SCRIPT_DIR}/..
    fi

    cd ${MODELS_SOURCE_DIR}/mmpose
    # miminum requirments - no onnx, etc.
    pip install -r requirements.txt
    pip3 install -v -e .

    mim install mmdet==3.1.0
    
    # during the installation isort version might be overwritten.
    # hence make sure back to the isort=5.13.2
    pip install isort==5.13.2

    if [ "${DOWNLOAD_WEIGHTS}" == "True" ]; then
        if [ -z "$(ls -A ${MODELS_WEIGHT_DIR}/mmpose)" ]; then
            echo
            echo "Downloading RTMO model weights"
            echo

            # RTMO: Towards High-Performance One-Stage Real-Time Multi-Person Pose Estimation
            WEIGHT_DIR="${MODELS_WEIGHT_DIR}/mmpose/rtmo_coco"
            mkdir -p ${WEIGHT_DIR}
            wget -nc https://download.openmmlab.com/mmpose/v1/projects/rtmo/rtmo-l_16xb16-600e_coco-640x640-516a421f_20231211.pth -P ${WEIGHT_DIR}
        else
            echo
            echo "MMPOSE-RTMO Weights directory is not empty, using existing models"
            echo
        fi
    fi
    # back to project root
    cd ${SCRIPT_DIR}/..
}

main "$@"