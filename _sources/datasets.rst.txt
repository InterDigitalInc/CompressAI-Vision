.. _dataset:

MPEG-VCM Datasets
=================

MPEG-VCM working group defines several evaluation datasets.

1. OIV6-MPEG
------------

Custom-modified `OpenImageV6 dataset <https://storage.googleapis.com/openimages/web/factsfigures_v6.html>`_:
hand-picked images and modified annotations in a custom format.

Required input files (don't worry - bundled with CompressAI-Vision):

.. code-block:: text

    detection_validation_5k_bbox.csv
    detection_validation_input_5k.lst
    detection_validation_labels_5k.csv
    segmentation_validation_bbox_5k.csv
    segmentation_validation_input_5k.lst
    segmentation_validation_labels_5k.csv
    segmentation_validation_masks_5k.csv

Commands to import:

.. code-block:: bash

    compressai-vision import-custom --dataset-type=oiv6-mpeg-v1

You can also define an additional argument ``--datadir=`` to indicate where the OpenImageV6 dataset is downloaded (by default to ``~/fiftyone``).

Final fiftyone dataset names:

- ``oiv6-mpeg-detection-v1``
- ``oiv6-mpeg-segmentation-v1``

2. TVD
------
Aka. Tencent Video Dataset.  Videos and images.  Annotations come in
a custom format identical to (1).

Download these files from `TVD page <https://multimedia.tencent.com/resources/tvd>`_:

.. code-block:: text

    TVD_Object_Detection_Dataset_and_Annotations.zip  # object detection annotations & images in a tar file
    TVD_Instance_Segmentation_Annotations.zip  # segmentations annotations & segmasks
    TVD.zip  # videos where the images were taken from - in fact, not required for using the datasets
    TVD_Object_Tracking_Dataset_and_Annotations.zip  # object tracking videos & annotations

After unpacking everything, you should have the following input directory/file structure
(call it ``/path/to/dir``):

.. code-block:: text

    TVD_Object_Detection_Dataset_And_Annotations/ [IMAGE DETECTION]
        tvd_object_detection_dataset/ [IMAGES]
        tvd_detection_validation_bbox.csv
        tvd_detection_validation_labels.csv

    tvd_segmentation_validation_bbox.csv
    tvd_segmentation_validation_labels.csv
    tvd_segmentation_validation_masks.csv
    tvd_validation_masks/ [VALIDATION MASK IMAGES]

    TVD_object_tracking_dataset_and_annotations/ [OBJECT TRACKING VIDEOS]
        TVD-01.mp4
        TVD-02.mp4
        TVD-03.mp4
        TVD-01/
            gt/gt.txt [ANNOTATIONS]
        TVD-02/
            gt/gt.txt [ANNOTATIONS]
        TVD-03/
            gt/gt.txt [ANNOTATIONS]

Commands to import:

.. code-block:: bash

    compressai-vision import-custom --dataset-type=tvd-object-tracking-v1 \
    --dir=/path/to/dir/TVD_object_tracking_dataset_and_annotations
    
    compressai-vision import-custom --dataset-type=tvd-image-v1 --dir=/path/to/dir

Final fiftyone dataset names:

- ``tvd-image-detection-v1``
- ``tvd-image-segmentation-v1``
- ``tvd-object-tracking-v1``


3. SFU-HW-Objects-V1
--------------------

Dataset consisting of raw YUV videofiles with video metadata in the filename.  Annotations
come in per-frame text files.

Download following input files from `here <https://www.frdr-dfdr.ca/repo/dataset/59931535-9ffd-4cc3-a3c2-4b06d06603d1>`_:

.. code-block:: text

    SFU-HW-Objects-v1.zip

After unpacking, following input directory/file structure (call it ``/path/to/dir``):

.. code-block:: text

    ClassA/
        PeopleOnStreet_2560x1600_30_crop.yuv
        Traffic_2560x1600_30_crop.yuv
        Annotations/
            PeopleOnStreet/
            Traffic/
    ClassB/
        ...
        ...
    ...
    ...

You must get the ``.yuv`` from someplace else & put the in-place as described above.

Commands to import:

.. code-block:: bash

    compressai-vision import-custom --dataset-type=sfu-hw-objects-v1 --dir=/path/to/dir

Final fiftyone dataset names:

- ``sfu-hw-objects-v1``


5. FLIR-MPEG
------------

Use the zipfile provided by the mpeg-vcm working group.  
After unpacking, you have the following input directory/file structure (call it ``/path/to/dir``):

.. code-block:: text

    ├── anchor_results
    │   ├── FLIR_anchor_vtm12_bitdepth10.xlsx
    │   └── VCM-reporting-template-FLIR_vtm12_d10.xlsm
    ├── dataset
    │   ├── coco_format_json_annotation
    │   │   ├── FLIR_val_thermal_coco_format_jpg.json
    │   │   ├── FLIR_val_thermal_coco_format_png.json
    │   │   └── Two files differ only in image file format whithin the file, and the rest are the same..txt
    │   ├── fine_tuned_model
    │   │   └── model_final.pth
    │   └── thermal_images [300 entries exceeds filelimit, not opening dir]
    ├── mAP_coco.py
    └── Readme.txt

Commands to import:

.. code-block:: bash

    compressai-vision import-custom --dataset-type=flir-mpeg-v1 --dir=/path/to/dir

Final fiftyone dataset names:

- ``flir-mpeg-detection-v1``


Other Datasets
==============

6. FLIR
-------

Nightime and infrared images.

Download following input files from `here <https://adas-dataset-v2.flirconservator.com/#downloadguide>`_:

.. code-block:: text
    
    FLIR_ADAS_v2.zip

After unpacking, you have the following input directory/file structure (call it ``/path/to/dir``):

.. code-block:: text

    rgb_to_thermal_vid_map.json
    images_rgb_train/
        coco_annotation_counts.tsv
        coco_annotation_counts.txt
        coco.json # Annotations in COCO format
        index.json
        data/ [IMAGES]
    images_rgb_val/
        ...
    images_thermal_train/
        ...
    images_thermal_val/
        ...
    video_rgb_test/
        ...
    video_thermal_test/
        ...

Commands to import:

.. code-block:: bash

    compressai-vision import-custom --dataset-type=flir-image-rgb-v1 --dir=/path/to/dir

Final fiftyone dataset names:

- ``flir-image-rgb-v1``

