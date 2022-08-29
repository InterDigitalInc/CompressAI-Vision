"""cli.py : Command-line interface tools for compressai-vision

* Copyright: 2022 InterDigital
* Authors  : Sampsa Riikonen
* Date     : 2022
* Version  : 0.1

This file is part of compressai-vision libraru
"""
# from compressai_vision import constant
from compressai_vision.local import AppLocalDir
from compressai_vision.tools import quickLog, confLogger, pathExists
#
import os, sys
# fiftyone
import fiftyone as fo
import fiftyone.zoo as foz
# compressai_vision

def main():
    try:
        import torch
    except ModuleNotFoundError:
        print("\nPYTORCH NOT INSTALLED\n")
        sys.exit(2)
    try:
        import detectron2
    except ModuleNotFoundError:
        print("\nDETECTRON2 NOT INSTALLED\ŋ")
        sys.exit(2)

    try:
        import compressai
    except ModuleNotFoundError:
        print("\nCOMPRESSAI NOT INSTALLED")
        sys.exit(2)

    print("\n*** TORCH, CUDA, DETECTRON2, COMPRESSAI ***")
    print("torch version       :", torch.__version__)
    print("cuda version        :", torch.version.cuda)
    print("detectron2 version  :", detectron2.__version__)
    print("compressai version  :", compressai.__version__)

    print("\n*** CHECKING GPU AVAILABILITY ***")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print("device              :", device)

    print("\n*** TESTING FFMPEG ***")
    c=os.system("ffmpeg -version")
    if c>0:
        print("\nRUNNING FFMPEG FAILED\n")
    #
    print("\n*** DATASETS ***")
    print("datasets currently registered into fiftyone")
    print("name, length, first sample path")
    for name in fo.list_datasets():
        dataset = fo.load_dataset(name)
        n = len(dataset)
        if n>0:
            sample=dataset.first()
            p=os.path.sep.join(sample["filepath"].split(os.path.sep)[:-1])
        else:
            p="?"
        print("%s, %i, %s" % (name, len(dataset), p))
    print()
