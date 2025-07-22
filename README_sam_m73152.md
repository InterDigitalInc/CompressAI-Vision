###
## Introduction 

This package is used for an input document m73152 - Introducing Sam - Segment Anything model. It is a modified version of CompressAI-Vision (version 1.1.14, tag 11accd67a7fe19f134d8f8b5564d4fa16d6d2389). Compare with the original CompressAI-Vision package, this package includes a new model, a new dataset type, and a new datacatalog specially designed for use with Sam.  

Dataset:
Dataset for testing is compressed and available at the MPEG content server: 
    mpegcontent@content.mpeg.expert:~/MPEG-AI/Part-4 FCM/dataset/samtest.tar.gz

The "samtest" sataset contains: 
    An "images" folder contains images for test, sourced from the FCM segmentation task, 
    A "annotations" folder containing segmentation ground truth in a json file,
    A "promts" folder with prompt file for each image. 

Prompt files are in text format. Each prompt file shares its name as its corresponding image, and contains one line for one prompt point belonging to one object in the image.



Pretrained weights for Sam can be found at: 
    https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth


Run the test
To run the test, use a command line in the format below: 

'''
bash scripts/evaluation/default_vision_performances_sam_withVTM.sh [mode] [TESTDATA_DIR] [device] [QP]
'''

Where:

scripts/evaluation/default_vision_performances_sam_withVTM.sh: is a script was modified from scripts/evaluation/default_vision_performances.sh script and used for the Sam test with inner codec of VTM version 23.3. The "INNER_CODEC_PATH" from this script may need to be changed to the actual path of the VTM will be used.

mode: can be split mode (mode=compressai-split-inference) or remote mode (mode=compressai-remote-inference). This experiment uses "compressai-split-inference" mode.

TESTDATA_DIR: directory to the dataset. For this experiment, the "samtest" dataset folder should place in TESTDATA_DIR. 

device: name of device, it can be "cpu" or "cuda" if cpu, or gpu is used respectively.


To run the test with four different QPs as used in this contribution, follow the belows command lines: 

'''
cd ./build/add_sam 

bash scripts/evaluation/default_vision_performances_sam_withVTM.sh compressai-split-inference {TESTDATA_DIR} cpu 20

bash scripts/evaluation/default_vision_performances_sam_withVTM.sh compressai-split-inference {TESTDATA_DIR} cpu 23

bash scripts/evaluation/default_vision_performances_sam_withVTM.sh compressai-split-inference {TESTDATA_DIR} cpu 24

bash scripts/evaluation/default_vision_performances_sam_withVTM.sh compressai-split-inference {TESTDATA_DIR} cpu 26
'''

Replace the {TESTDATA_DIR} by the directory to the dataset "samtest".
