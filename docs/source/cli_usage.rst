.. _cli:

Command line usage
==================

compressai-vision-info
----------------------

Shows information about your the installed software stack, library
versions and registered databases.

compressai-nokia-auto-import
----------------------------

Provided Nokia's input files, namely:

::

    detection_validation_input_5k.lst
    detection_validation_5k_bbox.csv
    detection_validation_labels_5k.csv
    segmentation_validation_input_5k.lst
    segmentation_validation_bbox_5k.csv
    segmentation_validation_labels_5k.csv
    segmentation_validation_masks_5k.csv

the command will:

1. Download Images
2. Perform input file conversion

Step (1) will use FiftyOne to download necessary images and segmentation masks from the
OpenImageV6 dataset.

Step (2) will convert Nokia's detection and segmentation annotations into proper
OpenImageV6 format and import (register) them into FiftyOne.

Type the command in terminal for more information

compressai-vision
-----------------

This command gives more control over steps (1) and (2).  After those steps, you can
run the whole evaluation pipeline using Detectron2:

::

    Image dataset --> Encoding --> calculate bitrate --> Decoding --> Detectron2 predictor

Type this command in your terminal for more information

::

    compressai-vision manual

Here are some example commands you might want to try 
(from the proper virtualenvironment or docker container, of course):

*OpenImageV6 evaluation for the Nokia 5K dataset*

::

    compressai-vision detectron2_eval --y --name=nokia-detection \
    --model=COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x.yaml \
    --output=eval5K.json

*OpenImageV6 evaluation for the Nokia 5K dataset with CompressAI to produce mAP(bpp) curve*

::

    compressai-vision detectron2_eval --y --name=nokia-detection \
    --model=COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x.yaml \
    --compressai=bmshj2018_factorized \
    --qpars=1,2,3,4,5,6,7,8 --output=eval5Kqp.json

*Test a single (mAP, bpp) point with VTM using a test/dummy database*

::

    compressai-vision detectron2_eval --y --debug \
    --name=nokia-detection-dummy \
    --model=COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x.yaml \
    --vtm --vtm_dir=path/to/VVCSoftware_VTM/bin --qpars=47 --output=vtm_test.json

