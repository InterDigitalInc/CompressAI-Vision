## FROM ufoym/deepo
## based on this excellent Docker image: https://github.com/ufoym/deepo#GPU
#
## WARNING: the very first step!  check with "pip3 install tensorflow-gpu==2.2.0" to see what tf versions you have available!
## after that, proceed in picking up the correct nvidia image with compatible cuda and cudnn versions
## tensorflow compatible versions here: https://www.tensorflow.org/install/source#gpu
## tags are here: https://hub.docker.com/r/nvidia/cuda/tags
##
# FROM nvidia/cuda:10.2-cudnn7-runtime-ubuntu18.04
## ..there's something wrong with that image... too "low-level".  Let's use this instead:
## FROM tensorflow/tensorflow:2.3.0-gpu
## ..that's based on ubuntu 18 LTS (old)
## this is more recent (ubuntu 20.04 focal) & has CUDA 11.2:
FROM tensorflow/tensorflow:2.9.1-gpu

USER root
ENV PATH="/usr/local/cuda/bin:$PATH"

RUN DEBIAN_FRONTEND=noninteractive apt-get update
RUN DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends apt-utils

## opencv requires libsm6 libxext6 libxrender-dev, vtm compilation needs build-essential, cmake
RUN DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
  libsm6 libxext6 libxrender-dev tree git build-essential wget tar cmake ffmpeg

## tracking the vtm dependencies.. needs cmake .. what else?
RUN DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends cmake

## download & compile VTM-12.0
RUN mkdir -p /root/vtm
WORKDIR /root/vtm
RUN wget https://vcgit.hhi.fraunhofer.de/jvet/VVCSoftware_VTM/-/archive/VTM-12.0/VVCSoftware_VTM-VTM-12.0.tar.gz
RUN tar xvf VVCSoftware_VTM-VTM-12.0.tar.gz
WORKDIR /root/vtm/VVCSoftware_VTM-VTM-12.0/build
RUN cmake .. -DCMAKE_BUILD_TYPE=Release
RUN make -j

## python stuff
RUN pip3 install --upgrade pip

RUN pip3 install \
  fiftyone \
  pycocotools \
  jupyter \
  ipython \
  torch==1.8.2 \
  torchvision==0.9.2 \
  pytorch-msssim \
  --extra-index-url https://download.pytorch.org/whl/lts/1.8/cu111
## see torch & torchvision compat here:
## https://pypi.org/project/torchvision/

## detectron2 & torch & cuda compat here:
## https://detectron2.readthedocs.io/en/latest/tutorials/install.html
RUN python3 -m pip install detectron2==0.4 -f \
  https://dl.fbaipublicfiles.com/detectron2/wheels/cu111/torch1.8/index.html

RUN pip3 install opencv-python==4.2.0.34 \
  progressbar \
  tqdm \
  PyYAML \
  tensorboard \
  Pillow \
  scikit-image

RUN pip3 install compressai
RUN pip3 install -vvv --upgrade git+https://github.com/InterDigitalInc/CompressAI-Vision.git

ENV PYTHONUNBUFFERED=1
ENV VTM_DIR=/root/vtm/VVCSoftware_VTM-VTM-12.0
WORKDIR /root

RUN python3 -c 'import torch, detectron2; print("torch:", torch.__version__, "/ cuda:", torch.version.cuda, "/ detectron2:", detectron2.__version__)'
RUN ffmpeg -version
