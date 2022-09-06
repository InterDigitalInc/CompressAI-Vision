#!/bin/bash
## installing these in correct order seems to be important
## due to versioning resons..
pip3 install -U pip
pip3 install fiftyone jupyter ipython
pip3 install torch==1.9.1 torchvision==0.10.1 pytorch-msssim
python3 -m pip install detectron2 -f https://dl.fbaipublicfiles.com/detectron2/wheels/cu102/torch1.9/index.html
pip3 install pybind11
pip3 install compressai
## WARNING TODO: this wont work until CompressAI-Vision becomes all public:
echo
echo INSTALLING COMPRESSAI-VISION DIRECTLY FROM PUBLIC REPO
echo IF YOU DONT WANT THIS, JUST PRESS CTRL-C
echo
pip3 install --upgrade git+https://github.com/InterDigitalInc/CompressAI-Vision.git
