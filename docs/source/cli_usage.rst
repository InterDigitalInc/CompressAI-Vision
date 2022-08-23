.. _cli:

Command line usage
==================

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

Type the command in terminal for more information
