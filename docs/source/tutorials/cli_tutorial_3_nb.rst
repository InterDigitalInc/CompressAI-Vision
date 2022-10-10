In this chapter you will learn:

-  to import the mpeg-vcm datasets (from the mpeg-vcm working group)
-  running evaluation on the mpeg-vcm datasets

The mpeg-vcm working group defines a specially prepared custom datasets
(subset of OpenImageV6) for evaluating the performance of your
deep-learning de/compression algorithm.

The tricky part is importing all that data into fiftyone. Once we have
done that, we can use the CLI tools to evaluate the de/compression model
with the mpeg-vcm defined pipeline, i.e.:

::

   mpeg-vcm custom dataset --> compression and decompression --> Detectron2 predictor --> mAP

The CLI tools have a subcommand ``mpeg-vcm-auto-import`` that downloads
necessary images from the open images dataset and prepares the
annotations according to mpeg-vcm working group specifications, so the
only thing you need to do to get started, is simply to type

::

   compressai-vision mpeg-vcm-auto-import

After running that “wizard”, you should have the mpeg-vcm datasets
registered into fiftyone (``mpeg-vcm-detection`` etc. datasets):

.. code:: ipython3

    !compressai-vision list


.. parsed-literal::

    importing fiftyone
    fiftyone imported
    
    datasets currently registered into fiftyone
    name, length, first sample path
    mpeg-vcm-detection, 5000, /home/sampsa/fiftyone/mpeg-vcm-detection/data
    mpeg-vcm-detection-dummy, 1, /home/sampsa/fiftyone/mpeg-vcm-detection/data
    mpeg-vcm-segmentation, 5000, /home/sampsa/fiftyone/mpeg-vcm-segmentation/data
    open-images-v6-validation, 8189, /home/sampsa/fiftyone/open-images-v6/validation/data
    quickstart, 200, /home/sampsa/fiftyone/quickstart/data


Now we can continue by evaluating the datasets agains a compressai
model, like we did in chapter 1. Before that, let’s take a closer look
at the dataset ``mpeg-vcm-detection``:

.. code:: ipython3

    !compressai-vision show --dataset-name=mpeg-vcm-detection


.. parsed-literal::

    importing fiftyone
    fiftyone imported
    
    dataset info:
    Name:        mpeg-vcm-detection
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
    
    test-loading first image from /home/sampsa/fiftyone/mpeg-vcm-detection/data/0001eeaf4aed83f9.jpg
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

.. code:: ipython3

    !compressai-vision detectron2-eval --y --dataset-name=mpeg-vcm-detection \\
    --slice=0:2 \\
    --gt-field=detections \\
    --eval-method=open-images \\
    --progressbar \\
    --output=detectron2_mpeg_vcm.json \\
    --model=COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x.yaml


.. parsed-literal::

    importing fiftyone
    fiftyone imported
    WARNING: using a dataset slice instead of full dataset: SURE YOU WANT THIS?
    
    Using dataset          : mpeg-vcm-detection
    Dataset tmp clone      : detectron-run-sampsa-mpeg-vcm-detection-2022-10-07-16-10-22-138077
    Image scaling          : 100
    WARNING: Using slice   : 0:2
    Number of samples      : 2
    Torch device           : cpu
    Detectron2 model       : COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x.yaml
    Model was trained with : coco_2017_train
    ** Evaluation without Encoding/Decoding **
    Ground truth data field name
                           : detections
    Eval. results will be saved to datafield
                           : detectron-predictions
    Evaluation protocol    : open-images
    Progressbar            : True
    WARNING: progressbar enabled --> disabling normal progress print
    Print progress         : 0
    Output file            : detectron2_mpeg_vcm.json
    Peek model classes     :
    ['airplane', 'apple', 'backpack', 'banana', 'baseball bat'] ...
    Peek dataset classes   :
    ['airplane', 'person'] ...
    cloning dataset mpeg-vcm-detection to detectron-run-sampsa-mpeg-vcm-detection-2022-10-07-16-10-22-138077
    instantiating Detectron2 predictor
    /home/sampsa/silo/interdigital/venv_all/lib/python3.8/site-packages/torch/_tensor.py:575: UserWarning: floor_divide is deprecated, and will be removed in a future version of pytorch. It currently rounds toward 0 (like the 'trunc' function NOT 'floor'). This results in incorrect rounding for negative values.
    To keep the current behavior, use torch.div(a, b, rounding_mode='trunc'), or for actual floor division, use torch.div(a, b, rounding_mode='floor'). (Triggered internally at  ../aten/src/ATen/native/BinaryOps.cpp:467.)
      return torch.floor_divide(self, other)
     100% |███████████████████████████████████████████████████████████████████| 2/2 error: number of pixels sum < 1
    Ignoring unsupported parameters {'compute_mAP'\} for <class 'fiftyone.utils.eval.openimages.OpenImagesEvaluationConfig'>
    Evaluating detections...
     100% |███████████| 2/2 [38.8ms elapsed, 0s remaining, 51.5 samples/s] 
    deleting tmp database detectron-run-sampsa-mpeg-vcm-detection-2022-10-07-16-10-22-138077
    
    HAVE A NICE DAY!
    


Next we create two points on the mAP(bbp) curve for the compressai
pre-trained ``bmshj2018_factorized`` model:

.. code:: ipython3

    !compressai-vision detectron2-eval --y --dataset-name=mpeg-vcm-detection \\
    --slice=0:2 \\
    --gt-field=detections \\
    --eval-method=open-images \\
    --progressbar \\
    --qpars=1,2 \\
    --compressai-model-name=bmshj2018_factorized \\
    --output=detectron2_mpeg_vcm_qpars.json \\
    --model=COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x.yaml


.. parsed-literal::

    importing fiftyone
    fiftyone imported
    WARNING: using a dataset slice instead of full dataset: SURE YOU WANT THIS?
    
    Using dataset          : mpeg-vcm-detection
    Dataset tmp clone      : detectron-run-sampsa-mpeg-vcm-detection-2022-10-07-16-17-12-468267
    Image scaling          : 100
    WARNING: Using slice   : 0:2
    Number of samples      : 2
    Torch device           : cpu
    Detectron2 model       : COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x.yaml
    Model was trained with : coco_2017_train
    Using compressai model : bmshj2018_factorized
    Quality parameters     : [1, 2]
    Ground truth data field name
                           : detections
    Eval. results will be saved to datafield
                           : detectron-predictions
    Evaluation protocol    : open-images
    Progressbar            : True
    WARNING: progressbar enabled --> disabling normal progress print
    Print progress         : 0
    Output file            : detectron2_mpeg_vcm_qpars.json
    Peek model classes     :
    ['airplane', 'apple', 'backpack', 'banana', 'baseball bat'] ...
    Peek dataset classes   :
    ['airplane', 'person'] ...
    cloning dataset mpeg-vcm-detection to detectron-run-sampsa-mpeg-vcm-detection-2022-10-07-16-17-12-468267
    instantiating Detectron2 predictor
    
    QUALITY PARAMETER:  1
    /home/sampsa/silo/interdigital/venv_all/lib/python3.8/site-packages/torch/_tensor.py:575: UserWarning: floor_divide is deprecated, and will be removed in a future version of pytorch. It currently rounds toward 0 (like the 'trunc' function NOT 'floor'). This results in incorrect rounding for negative values.
    To keep the current behavior, use torch.div(a, b, rounding_mode='trunc'), or for actual floor division, use torch.div(a, b, rounding_mode='floor'). (Triggered internally at  ../aten/src/ATen/native/BinaryOps.cpp:467.)
      return torch.floor_divide(self, other)
     100% |███████████████████████████████████████████████████████████████████| 2/2 Ignoring unsupported parameters {'compute_mAP'\} for <class 'fiftyone.utils.eval.openimages.OpenImagesEvaluationConfig'>
    Evaluating detections...
     100% |███████████| 2/2 [23.9ms elapsed, 0s remaining, 83.7 samples/s] 
    
    QUALITY PARAMETER:  2
     100% |███████████████████████████████████████████████████████████████████| 2/2 Ignoring unsupported parameters {'compute_mAP'\} for <class 'fiftyone.utils.eval.openimages.OpenImagesEvaluationConfig'>
    Evaluating detections...
     100% |███████████| 2/2 [26.6ms elapsed, 0s remaining, 75.2 samples/s] 
    deleting tmp database detectron-run-sampsa-mpeg-vcm-detection-2022-10-07-16-17-12-468267
    
    HAVE A NICE DAY!
    


Again, for an actual production run, you would remove the ``--slice``
argument. You can run all quality points (bpp values) in a single run,
say by defining ``--qpars=1,2,3,4,5,6,7,8``, or if you want to
parallelize, send the same command to your queue system several times,
each time with a different quality parameter values,
i.e. \\ ``--qpars=1``, ``--qpars=2``, etc.

