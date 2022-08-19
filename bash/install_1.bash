#!/bin/bash
## installing these in correct order seems to be important
## due to versioning resons..
pip3 install fiftyone pycocotools jupyter
pip3 install torch==1.9.1 torchvision==0.10.1 pytorch-msssim
python3 -m pip install detectron2 -f https://dl.fbaipublicfiles.com/detectron2/wheels/cu102/torch1.9/index.html
pip3 install compressai
pip3 install --upgrade git+git://https://github.com/InterDigitalInc/CompressAI-Vision
