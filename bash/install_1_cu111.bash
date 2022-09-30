#!/bin/bash
# WARNING: keep consistent with ../compressai_vision/__init__.py:
FO_VERSION="0.16.6" 
python3 -c "import sys; assert(sys.version_info.major>=3); assert(sys.version_info.minor>=8)"
if [[ $? -gt 0 ]] 
then
    echo
    echo "Your python version needs to be >=3.8"
    echo
    exit 2
fi
## installing these in correct order seems to be important
## due to versioning resons..
pip3 install -U pip
# word of warning here: depending on your python version, a different fiftyone
# version might get installed, so we fix explicitly here the fiftyone version.  you need python3.8+ for this version
pip3 install fiftyone==$FO_VERSION jupyter ipython
pip install torch==1.10.1+cu111 torchvision==0.11.2+cu111 -f https://download.pytorch.org/whl/torch_stable.html
pip install pytorch-msssim
python -m pip install detectron2 -f  https://dl.fbaipublicfiles.com/detectron2/wheels/cu111/torch1.10/index.html
pip3 install pybind11
pip3 install compressai
## WARNING TODO: this wont work until CompressAI-Vision becomes all public:
echo
echo INSTALLING COMPRESSAI-VISION DIRECTLY FROM PUBLIC REPO
echo IF YOU DONT WANT THIS, JUST PRESS CTRL-C
echo
pip3 install --upgrade git+https://github.com/InterDigitalInc/CompressAI-Vision.git
