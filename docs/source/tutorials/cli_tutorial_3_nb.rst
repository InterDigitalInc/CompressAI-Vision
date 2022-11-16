In this chapter you will learn:

-  to import the mpeg-vcm working-group custom datasets
-  running evaluation on dataset

The mpeg-vcm working group defines several custom datasets for
evaluating the performance of your deep-learning de/compression
algorithm. For more details, please see the Datasets section of the
documentation.

The tricky part is importing all that data into fiftyone. Once we have
done that, we can use the CLI tools to evaluate the de/compression model
with the mpeg-vcm defined pipeline, i.e.:

::

   mpeg-vcm custom dataset --> compression and decompression --> Detectron2 predictor --> mAP

All the datasets can be download and/or registered into fiftyone with
the ``compressai-vision import-custom`` command.

For example, after running
``compressai-vision import-custom oiv6-mpeg-v1`` you will have the
following datasets:

-  ``oiv6-mpeg-detection-v1``
-  ``oiv6-mpeg-segmentation-v1``

.. code:: ipython3

    compressai-vision list


.. code-block:: text

    importing fiftyone
    fiftyone imported
    
    datasets currently registered into fiftyone
    name, length, first sample path
    flir-image-rgb-v1, 10318, /media/sampsa/4d0dff98-8e61-4a0b-a97e-ceb6bc7ccb4b/datasets/flir/images_rgb_train/data
    oiv6-mpeg-detection-v1, 5000, /home/sampsa/fiftyone/oiv6-mpeg-detection-v1/data
    oiv6-mpeg-detection-v1-dummy, 1, /home/sampsa/fiftyone/oiv6-mpeg-detection-v1/data
    oiv6-mpeg-segmentation-v1, 5000, /home/sampsa/fiftyone/oiv6-mpeg-segmentation-v1/data
    open-images-v6-validation, 8189, /home/sampsa/fiftyone/open-images-v6/validation/data
    quickstart, 200, /home/sampsa/fiftyone/quickstart/data
    quickstart-video, 10, /home/sampsa/fiftyone/quickstart-video/data
    sfu-hw-objects-v1, 2, /home/sampsa/silo/interdigital/mock/SFU-HW-Objects-v1/ClassC/Annotations/BasketballDrill
    tvd-image-detection-v1, 167, /media/sampsa/4d0dff98-8e61-4a0b-a97e-ceb6bc7ccb4b/datasets/tvd/TVD_images_detection_v1/data
    tvd-image-segmentation-v1, 167, /media/sampsa/4d0dff98-8e61-4a0b-a97e-ceb6bc7ccb4b/datasets/tvd/TVD_images_segmentation_v1/data
    tvd-object-tracking-v1, 3, /media/sampsa/4d0dff98-8e61-4a0b-a97e-ceb6bc7ccb4b/datasets/tvd/TVD_object_tracking_dataset_and_annotations


Now we can continue by evaluating the datasets agains a compressai
model, like we did in chapter 1. Before that, let’s take a closer look
at the dataset ``oiv6-mpeg-detection-v1``:

.. code:: bash

    compressai-vision show --dataset-name=oiv6-mpeg-detection-v1


.. code-block:: text

    importing fiftyone
    fiftyone imported
    
    dataset info:
    Name:        oiv6-mpeg-detection-v1
    Media type:  image
    Num samples: 5000
    Persistent:  True
    Tags:        []
    Sample fields:
        id:              fiftyone.core.fields.ObjectIdField
        filepath:        fiftyone.core.fields.StringField
        tags:            fiftyone.core.fields.ListField(fiftyone.core.fields.StringField)
        metadata:        fiftyone.core.fields.EmbeddedDocumentField(fiftyone.core.metadata.ImageMetadata)
        positive_labels: fiftyone.core.fields.EmbeddedDocumentField(fiftyone.core.labels.Classifications)
        negative_labels: fiftyone.core.fields.EmbeddedDocumentField(fiftyone.core.labels.Classifications)
        detections:      fiftyone.core.fields.EmbeddedDocumentField(fiftyone.core.labels.Detections)
        open_images_id:  fiftyone.core.fields.StringField
    
    test-loading first image from /home/sampsa/fiftyone/oiv6-mpeg-detection-v1/data/0001eeaf4aed83f9.jpg
    loaded image with dimensions (447, 1024, 3) ok


Detection data ground truths (bounding boxes) in each sample are in the
field ``detections``, so we need to use ``--gt-field=detections``.
Evaluation method for mAP is the OpenImagesV6 protocol, so we use
``--eval-method=open-images``. For a quick test run we just run the
evaluation with the two first images of the dataset with ``--slice=0:2``
(for an actual production run, remove it).

To get an mAP reference value (without any sort of de/compression), we
run crunch images through a Detectron2 predictor and compare to the
ground truths in field ``detections``:

.. code:: bash

    compressai-vision detectron2-eval --y --dataset-name=oiv6-mpeg-detection-v1 \
    --slice=0:2 \
    --gt-field=detections \
    --eval-method=open-images \
    --progressbar \
    --output=detectron2_mpeg_vcm.json \
    --model=COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x.yaml


.. code-block:: text

    importing fiftyone
    fiftyone imported
    WARNING: using a dataset slice instead of full dataset: SURE YOU WANT THIS?
    instantiating Detectron2 predictor 0 : COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x.yaml
    
    Using dataset          : oiv6-mpeg-detection-v1
    Dataset media type     : image
    Dataset tmp clone      : detectron-run-sampsa-oiv6-mpeg-detection-v1-2022-11-16-17-21-51-787050
    Keep tmp dataset?      : False
    Image scaling          : 100
    WARNING: Using slice   : 0:2
    Number of samples      : 2
    Torch device           : cpu
    === Vision Model #0 ====
    Detectron2 model       : COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x.yaml
    Model was trained with : coco_2017_train
    Eval. results will be saved to datafield
                           : detectron-predictions_v0
    Evaluation protocol    : open-images
    Peek model classes     :
    ['airplane', 'apple', 'backpack', 'banana', 'baseball bat'] ...
    Peek dataset classes   :
    ['airplane', 'person'] ...
    ** Evaluation without Encoding/Decoding **
    Ground truth data field name
                           : detections
    Progressbar            : True
    WARNING: progressbar enabled --> disabling normal progress print
    Print progress         : 0
    Output file            : detectron2_mpeg_vcm.json
    cloning dataset oiv6-mpeg-detection-v1 to detectron-run-sampsa-oiv6-mpeg-detection-v1-2022-11-16-17-21-51-787050
    /home/sampsa/silo/interdigital/venv_all/lib/python3.8/site-packages/torch/_tensor.py:575: UserWarning: floor_divide is deprecated, and will be removed in a future version of pytorch. It currently rounds toward 0 (like the 'trunc' function NOT 'floor'). This results in incorrect rounding for negative values.
    To keep the current behavior, use torch.div(a, b, rounding_mode='trunc'), or for actual floor division, use torch.div(a, b, rounding_mode='floor'). (Triggered internally at  ../aten/src/ATen/native/BinaryOps.cpp:467.)
      return torch.floor_divide(self, other)
     100% |███████████████████████████████████████████████████████████████████| 2/2 Evaluating detections...
     100% |███████████| 2/2 [24.9ms elapsed, 0s remaining, 80.3 samples/s] 
    deleting tmp database detectron-run-sampsa-oiv6-mpeg-detection-v1-2022-11-16-17-21-51-787050
    
    Done!
    


Next we create two points on the mAP(bbp) curve for the compressai
pre-trained ``bmshj2018_factorized`` model:

.. code:: bash

    compressai-vision detectron2-eval --y --dataset-name=oiv6-mpeg-detection-v1 \
    --slice=0:2 \
    --gt-field=detections \
    --eval-method=open-images \
    --progressbar \
    --qpars=1,2 \
    --compressai-model-name=bmshj2018-factorized \
    --output=detectron2_mpeg_vcm_qpars.json \
    --model=COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x.yaml


.. code-block:: text

    importing fiftyone
    fiftyone imported
    WARNING: using a dataset slice instead of full dataset: SURE YOU WANT THIS?
    instantiating Detectron2 predictor 0 : COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x.yaml
    
    Using dataset          : oiv6-mpeg-detection-v1
    Dataset media type     : image
    Dataset tmp clone      : detectron-run-sampsa-oiv6-mpeg-detection-v1-2022-11-16-17-28-02-372323
    Keep tmp dataset?      : False
    Image scaling          : 100
    WARNING: Using slice   : 0:2
    Number of samples      : 2
    Torch device           : cpu
    === Vision Model #0 ====
    Detectron2 model       : COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x.yaml
    Model was trained with : coco_2017_train
    Eval. results will be saved to datafield
                           : detectron-predictions_v0
    Evaluation protocol    : open-images
    Peek model classes     :
    ['airplane', 'apple', 'backpack', 'banana', 'baseball bat'] ...
    Peek dataset classes   :
    ['airplane', 'person'] ...
    Using compressai model : bmshj2018-factorized
    Quality parameters     : [1, 2]
    Ground truth data field name
                           : detections
    Progressbar            : True
    WARNING: progressbar enabled --> disabling normal progress print
    Print progress         : 0
    Output file            : detectron2_mpeg_vcm_qpars.json
    cloning dataset oiv6-mpeg-detection-v1 to detectron-run-sampsa-oiv6-mpeg-detection-v1-2022-11-16-17-28-02-372323
    
    QUALITY PARAMETER:  1
    /home/sampsa/silo/interdigital/venv_all/lib/python3.8/site-packages/torch/_tensor.py:575: UserWarning: floor_divide is deprecated, and will be removed in a future version of pytorch. It currently rounds toward 0 (like the 'trunc' function NOT 'floor'). This results in incorrect rounding for negative values.
    To keep the current behavior, use torch.div(a, b, rounding_mode='trunc'), or for actual floor division, use torch.div(a, b, rounding_mode='floor'). (Triggered internally at  ../aten/src/ATen/native/BinaryOps.cpp:467.)
      return torch.floor_divide(self, other)
     100% |███████████████████████████████████████████████████████████████████| 2/2 Evaluating detections...
     100% |███████████| 2/2 [15.2ms elapsed, 0s remaining, 131.9 samples/s] 
    
    QUALITY PARAMETER:  2
     100% |███████████████████████████████████████████████████████████████████| 2/2 Evaluating detections...
     100% |███████████| 2/2 [21.9ms elapsed, 0s remaining, 91.4 samples/s] 
    deleting tmp database detectron-run-sampsa-oiv6-mpeg-detection-v1-2022-11-16-17-28-02-372323
    
    Done!
    


Again, for an actual production run, you would remove the ``--slice``
argument. You can run all quality points (bpp values) in a single run,
say by defining ``--qpars=1,2,3,4,5,6,7,8``, or if you want to
parallelize, send the same command to your queue system several times,
each time with a different quality parameter values,
i.e. \ ``--qpars=1``, ``--qpars=2``, etc.

Again, and as explained in tutorial 1 you can visualize your dataset
with ``compressai-vision app`` command and compare ground-truths and
detections if you use ``--keep`` flag with the ``detectron2-eval``
command.

