In this tutorial chapter you will learn:

-  Checking the installed software stack with ``compressai-vision info``
-  Downloading datasets with ``compressai-vision download``
-  Evaluating datasets with ``compressai-vision detectron2-eval`` for
   creating mAP(bpp) curves

The command line interface (cli) has all the functionality for
evaluating your deep-learning compression algorithm against standardized
benchmarks.

The cli is accessed with the ``compressai-vision`` command that has
several subcommands for handling datasets, evaluating your models with
them and for generating plots. In detail:

-  ``compressai-vision -h`` gives you a short description of all
   commands
-  ``compressai-vision manual`` shows you a more thorough description
-  ``compressai-vision subcommand -h`` gives a detailed description of a
   certain subcommand

The very first subcommand you should try is ``info``. It gives you
information about the installed software stack, library versions and
registered datasets:

.. code:: ipython3

    compressai-vision info


.. parsed-literal::

    
    *** TORCH, CUDA, DETECTRON2, COMPRESSAI ***
    torch version       : 1.9.1+cu102
    cuda version        : 10.2
    detectron2 version  : 0.6
    compressai version  : 1.2.0.dev0
    
    *** CHECKING GPU AVAILABILITY ***
    device              : cpu
    
    *** TESTING FFMPEG ***
    ffmpeg version 4.2.7-0ubuntu0.1 Copyright (c) 2000-2022 the FFmpeg developers
    built with gcc 9 (Ubuntu 9.4.0-1ubuntu1~20.04.1)
    configuration: --prefix=/usr --extra-version=0ubuntu0.1 --toolchain=hardened --libdir=/usr/lib/x86_64-linux-gnu --incdir=/usr/include/x86_64-linux-gnu --arch=amd64 --enable-gpl --disable-stripping --enable-avresample --disable-filter=resample --enable-avisynth --enable-gnutls --enable-ladspa --enable-libaom --enable-libass --enable-libbluray --enable-libbs2b --enable-libcaca --enable-libcdio --enable-libcodec2 --enable-libflite --enable-libfontconfig --enable-libfreetype --enable-libfribidi --enable-libgme --enable-libgsm --enable-libjack --enable-libmp3lame --enable-libmysofa --enable-libopenjpeg --enable-libopenmpt --enable-libopus --enable-libpulse --enable-librsvg --enable-librubberband --enable-libshine --enable-libsnappy --enable-libsoxr --enable-libspeex --enable-libssh --enable-libtheora --enable-libtwolame --enable-libvidstab --enable-libvorbis --enable-libvpx --enable-libwavpack --enable-libwebp --enable-libx265 --enable-libxml2 --enable-libxvid --enable-libzmq --enable-libzvbi --enable-lv2 --enable-omx --enable-openal --enable-opencl --enable-opengl --enable-sdl2 --enable-libdc1394 --enable-libdrm --enable-libiec61883 --enable-nvenc --enable-chromaprint --enable-frei0r --enable-libx264 --enable-shared
    libavutil      56. 31.100 / 56. 31.100
    libavcodec     58. 54.100 / 58. 54.100
    libavformat    58. 29.100 / 58. 29.100
    libavdevice    58.  8.100 / 58.  8.100
    libavfilter     7. 57.100 /  7. 57.100
    libavresample   4.  0.  0 /  4.  0.  0
    libswscale      5.  5.100 /  5.  5.100
    libswresample   3.  5.100 /  3.  5.100
    libpostproc    55.  5.100 / 55.  5.100
    
    NOTICE: Using mongodb managed by fiftyone
    Be sure not to have extra mongod server(s) running on your system
    importing fiftyone..
    ..imported
    fiftyone version: 0.16.6
    
    *** DATABASE ***
    info about your connection:
    Database(MongoClient(host=['localhost:40083'], document_class=dict, tz_aware=False, connect=True, appname='fiftyone'), 'fiftyone')
    
    
    *** DATASETS ***
    datasets currently registered into fiftyone
    name, length, first sample path
    mpeg-vcm-detection, 5000, /home/sampsa/fiftyone/mpeg-vcm-detection/data
    mpeg-vcm-detection-dummy, 1, /home/sampsa/fiftyone/mpeg-vcm-detection/data
    mpeg-vcm-segmentation, 5000, /home/sampsa/fiftyone/mpeg-vcm-segmentation/data
    open-images-v6-validation, 8189, /home/sampsa/fiftyone/open-images-v6/validation/data
    


Another basic command is ``list`` that just shows you the registered
datasets:

.. code:: ipython3

    compressai-vision list


.. parsed-literal::

    importing fiftyone
    fiftyone imported
    
    datasets currently registered into fiftyone
    name, length, first sample path
    mpeg-vcm-detection, 5000, /home/sampsa/fiftyone/mpeg-vcm-detection/data
    mpeg-vcm-detection-dummy, 1, /home/sampsa/fiftyone/mpeg-vcm-detection/data
    mpeg-vcm-segmentation, 5000, /home/sampsa/fiftyone/mpeg-vcm-segmentation/data
    open-images-v6-validation, 8189, /home/sampsa/fiftyone/open-images-v6/validation/data


Datasets can be registered to and deregistered from fiftyone using the
``register`` and ``deregister`` subcommands, and downloaded and
registered directly from `fiftyone dataset
zoo <https://voxel51.com/docs/fiftyone/user_guide/dataset_zoo/datasets.html#dataset-zoo-quickstart>`__
with the ``download`` command. Let’s use ``download`` to get the
“quickstart” dataset:

.. code:: ipython3

    compressai-vision download --dataset-name=quickstart --y


.. parsed-literal::

    importing fiftyone
    fiftyone imported
    
    WARNING: downloading ALL images.  You might want to use the --lists option to download only certain images
    Using list files:     None
    Number of images:     ?
    Database name   :     quickstart
    Subname/split   :     None
    Target dir      :     None
    
    Dataset already downloaded
    Loading 'quickstart'
     100% |███████| 200/200 [3.0s elapsed, 0s remaining, 52.1 samples/s]       
    Dataset 'quickstart' created


Nice, we have ourselves a dataset to play with. A note: the ``--y``
switch makes the command to run in non-interactive mode. Let’s take a
closer look at the fields that the samples have in this datafield with
``show``:

.. code:: ipython3

    compressai-vision show --dataset-name=quickstart --y


.. parsed-literal::

    importing fiftyone
    fiftyone imported
    
    dataset info:
    Name:        quickstart
    Media type:  image
    Num samples: 200
    Persistent:  True
    Tags:        []
    Sample fields:
        id:           fiftyone.core.fields.ObjectIdField
        filepath:     fiftyone.core.fields.StringField
        tags:         fiftyone.core.fields.ListField(fiftyone.core.fields.StringField)
        metadata:     fiftyone.core.fields.EmbeddedDocumentField(fiftyone.core.metadata.ImageMetadata)
        ground_truth: fiftyone.core.fields.EmbeddedDocumentField(fiftyone.core.labels.Detections)
        uniqueness:   fiftyone.core.fields.FloatField
        predictions:  fiftyone.core.fields.EmbeddedDocumentField(fiftyone.core.labels.Detections)
    
    test-loading first image from /home/sampsa/fiftyone/quickstart/data/000880.jpg
    loaded image with dimensions (480, 640, 3) ok


Some fields of interests in each sample: ``filepath`` fields have the
path to the downloaded images, while ``ground_truth`` fields have the
ground-truth bounding boxes (“quickstart” dataset is a demo subset of
COCO).

Next we’ll crunch all the images in the dataset through a Detectron2
predictor and evaluate the results using the COCO evaluation protocol:
as a result, we’ll get a mAP accuracy for the Detectron2 model. Note
that we have to indicate the ground truth field with
``--gt-field=ground_truth``. Option ``--slice=0:2`` takes only the first
two samples from the dataset for this run: its only for debugging run,
so please feel free to remove it.

.. code:: ipython3

    compressai-vision detectron2-eval --y --dataset-name=quickstart \
    --slice=0:2 \
    --gt-field=ground_truth \
    --eval-method=coco \
    --progressbar \
    --output=detectron2_test.json \
    --model=COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x.yaml


.. parsed-literal::

    importing fiftyone
    fiftyone imported
    WARNING: using a dataset slice instead of full dataset: SURE YOU WANT THIS?
    
    Using dataset          : quickstart
    Dataset tmp clone      : detectron-run-sampsa-quickstart-2022-10-10-22-29-27-260938
    Image scaling          : 100
    WARNING: Using slice   : 0:2
    Number of samples      : 2
    Torch device           : cpu
    Detectron2 model       : COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x.yaml
    Model was trained with : coco_2017_train
    ** Evaluation without Encoding/Decoding **
    Ground truth data field name
                           : ground_truth
    Eval. results will be saved to datafield
                           : detectron-predictions
    Evaluation protocol    : coco
    Progressbar            : True
    WARNING: progressbar enabled --> disabling normal progress print
    Print progress         : 0
    Output file            : detectron2_test.json
    Peek model classes     :
    ['airplane', 'apple', 'backpack', 'banana', 'baseball bat'] ...
    Peek dataset classes   :
    ['bird', 'horse', 'person'] ...
    cloning dataset quickstart to detectron-run-sampsa-quickstart-2022-10-10-22-29-27-260938
    instantiating Detectron2 predictor
    /home/sampsa/silo/interdigital/venv_all/lib/python3.8/site-packages/torch/_tensor.py:575: UserWarning: floor_divide is deprecated, and will be removed in a future version of pytorch. It currently rounds toward 0 (like the 'trunc' function NOT 'floor'). This results in incorrect rounding for negative values.
    To keep the current behavior, use torch.div(a, b, rounding_mode='trunc'), or for actual floor division, use torch.div(a, b, rounding_mode='floor'). (Triggered internally at  ../aten/src/ATen/native/BinaryOps.cpp:467.)
      return torch.floor_divide(self, other)
     100% |███████████████████████████████████████████████████████████████████| 2/2 error: number of pixels sum < 1
    Evaluating detections...
     100% |███████████| 2/2 [9.5ms elapsed, 0s remaining, 211.5 samples/s] 
    Performing IoU sweep...
     100% |███████████| 2/2 [12.2ms elapsed, 0s remaining, 163.9 samples/s] 
    deleting tmp database detectron-run-sampsa-quickstart-2022-10-10-22-29-27-260938
    
    HAVE A NICE DAY!
    


Let’s see what we got:

.. code:: ipython3

    cat detectron2_test.json


.. parsed-literal::

    {
      "dataset": "quickstart",
      "gt_field": "ground_truth",
      "tmp datasetname": "detectron-run-sampsa-quickstart-2022-10-10-22-29-27-260938",
      "slice": "0:2",
      "model": "COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x.yaml",
      "codec": "",
      "qpars": null,
      "bpp": [
        -1
      ],
      "map": [
        0.5676567656765678
      ],
      "map_per_class": [
        {
          "bird": 0.30297029702970296,
          "horse": 0.5,
          "person": 0.9
        }
      ]
    }

Now we use again a Detectron2 predictor on our dataset. However, before
passing the images to Detectron2 model, they are first compressed and
decompressed by using a pre-trained compressai model with a quality
parameter 1 (``--qpars=1``).

We could evaluate for several quality parameters in serial by defining a
list, i.e: ``--qpars=1,2,3`` and in parallel by launching the command
separately for each particular value (say, for calculations in a
queue/grid system).

A scaling can be applied on the images, as defined by the mpeg-vcm
specifications (``--scale=100``). Again, remember to remove
``--slice=0:2`` for an actual run.

.. code:: ipython3

    compressai-vision detectron2-eval --y --dataset-name=quickstart \
    --slice=0:2 \
    --gt-field=ground_truth \
    --eval-method=coco \
    --scale=100 \
    --progressbar \
    --qpars=1 \
    --compressai-model-name=bmshj2018_factorized \
    --output=compressai_detectron2_test.json \
    --model=COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x.yaml


.. parsed-literal::

    importing fiftyone
    fiftyone imported
    WARNING: using a dataset slice instead of full dataset: SURE YOU WANT THIS?
    
    Using dataset          : quickstart
    Dataset tmp clone      : detectron-run-sampsa-quickstart-2022-10-10-22-29-49-246836
    Image scaling          : 100
    WARNING: Using slice   : 0:2
    Number of samples      : 2
    Torch device           : cpu
    Detectron2 model       : COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x.yaml
    Model was trained with : coco_2017_train
    Using compressai model : bmshj2018_factorized
    Quality parameters     : [1]
    Ground truth data field name
                           : ground_truth
    Eval. results will be saved to datafield
                           : detectron-predictions
    Evaluation protocol    : coco
    Progressbar            : True
    WARNING: progressbar enabled --> disabling normal progress print
    Print progress         : 0
    Output file            : compressai_detectron2_test.json
    Peek model classes     :
    ['airplane', 'apple', 'backpack', 'banana', 'baseball bat'] ...
    Peek dataset classes   :
    ['bird', 'horse', 'person'] ...
    cloning dataset quickstart to detectron-run-sampsa-quickstart-2022-10-10-22-29-49-246836
    instantiating Detectron2 predictor
    
    QUALITY PARAMETER:  1
    /home/sampsa/silo/interdigital/venv_all/lib/python3.8/site-packages/torch/_tensor.py:575: UserWarning: floor_divide is deprecated, and will be removed in a future version of pytorch. It currently rounds toward 0 (like the 'trunc' function NOT 'floor'). This results in incorrect rounding for negative values.
    To keep the current behavior, use torch.div(a, b, rounding_mode='trunc'), or for actual floor division, use torch.div(a, b, rounding_mode='floor'). (Triggered internally at  ../aten/src/ATen/native/BinaryOps.cpp:467.)
      return torch.floor_divide(self, other)
     100% |███████████████████████████████████████████████████████████████████| 2/2 Evaluating detections...
     100% |███████████| 2/2 [21.9ms elapsed, 0s remaining, 91.5 samples/s] 
    Performing IoU sweep...
     100% |███████████| 2/2 [30.0ms elapsed, 0s remaining, 66.8 samples/s] 
    deleting tmp database detectron-run-sampsa-quickstart-2022-10-10-22-29-49-246836
    
    HAVE A NICE DAY!
    


Let’s see what we got:

.. code:: ipython3

    cat compressai_detectron2_test.json


.. parsed-literal::

    {
      "dataset": "quickstart",
      "gt_field": "ground_truth",
      "tmp datasetname": "detectron-run-sampsa-quickstart-2022-10-10-22-29-49-246836",
      "slice": "0:2",
      "model": "COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x.yaml",
      "codec": "bmshj2018_factorized",
      "qpars": [
        1
      ],
      "bpp": [
        0.18178251121076233
      ],
      "map": [
        0.44477447744774484
      ],
      "map_per_class": [
        {
          "bird": 0.100990099009901,
          "horse": 0.3333333333333334,
          "person": 0.9
        }
      ]
    }

Which is a single point on the mAP(bpp) curve. Next you need to produce
some more points and then use ``plot`` subcommand. An explicit example
of that is given in the mpeg-vcm section of this tutorial.

