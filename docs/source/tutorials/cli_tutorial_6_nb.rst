In this chapter you will learn:

-  to generate and cache VTM-encoded bitstream
-  to create VTM baseline results

The subcommand ``vtm`` encodes images from your dataset with the VTM
program. VTM features state-of-the-art classical video encoding
techniques and it is used as a benchmark against your deep-learning
encoder’s efficiency. You need to download and compile the VTM software
yourself according to the instructions in the main documentation.

Why do we need a separate subcommand ``vtm`` for VTM encoding /
bitstream generation, instead of just using ``detectron2-eval`` on the
fly? i.e. for doing:

::

   Image --> VTM encoding --> VTM decoding --> Detectron2

That is because encoding performed by VTM is *very* CPU intensive task,
so it is something you don’t really want to repeat (encoding 5000 sample
images, depending on your qpars value, can take several days..!), so we
use the ``vtm`` subcommand to manage, encode and cache the VTM produced
bitstreams on disk.

Let’s generate some encoded bitstreams.

.. code:: ipython3

    compressai-vision vtm --y --dataset-name=oiv6-mpeg-detection-v1 \
    --slice=0:2 \
    --scale=100 \
    --progress=1 \
    --qpars=47 \
    --vtm_cache=/tmp/bitstreams \
    --vtm_dir={path_to_vtm_software}/bin \
    --vtm_cfg={path_to_vtm_software}/cfg/encoder_intra_vtm.cfg \
    --output=vtm_out.json


.. code-block:: text

    importing fiftyone
    fiftyone imported
    Reading vtm config from: /home/sampsa/silo/interdigital/VVCSoftware_VTM/cfg/encoder_intra_vtm.cfg
    WARNING: using a dataset slice instead of full dataset: SURE YOU WANT THIS?
    
    VTM bitstream generation
    WARNING: VTM USES CACHE IN /tmp/bitstreams
    Target dir             : /tmp/bitstreams
    Quality points/subdirs : [47]
    Using dataset          : oiv6-mpeg-detection-v1
    Image Scaling          : 100
    Using slice            : 0:2
    Number of samples      : 2
    Progressbar            : False
    Output file            : vtm_out.json
    Print progress         : 1
    
    QUALITY PARAMETER 47
    VTMEncoderDecoder - INFO - creating /tmp/bitstreams
    VTMEncoderDecoder - WARNING - creating bitstream /tmp/bitstreams/100/47/bin_0001eeaf4aed83f9 with VTMEncode from scratch
    sample:  1 / 2 tag: 0001eeaf4aed83f9
    VTMEncoderDecoder - WARNING - creating bitstream /tmp/bitstreams/100/47/bin_000a1249af2bc5f0 with VTMEncode from scratch
    sample:  2 / 2 tag: 000a1249af2bc5f0
    
    Done!
    


As you can see, bitstreams we’re generated and cached into
``/tmp/bitstreams/SCALE/QP``. Let’s see what happens if we run the exact
same command again:

.. code:: ipython3

    compressai-vision vtm --y --dataset-name=oiv6-mpeg-detection-v1 \
    --slice=0:2 \
    --scale=100 \
    --progress=1 \
    --qpars=47 \
    --vtm_cache=/tmp/bitstreams \
    --vtm_dir={path_to_vtm_software}/bin \
    --vtm_cfg={path_to_vtm_software}/cfg/encoder_intra_vtm.cfg \
    --output=vtm_out.json


.. code-block:: text

    importing fiftyone
    fiftyone imported
    Reading vtm config from: /home/sampsa/silo/interdigital/VVCSoftware_VTM/cfg/encoder_intra_vtm.cfg
    WARNING: using a dataset slice instead of full dataset: SURE YOU WANT THIS?
    
    VTM bitstream generation
    WARNING: VTM USES CACHE IN /tmp/bitstreams
    Target dir             : /tmp/bitstreams
    Quality points/subdirs : [47]
    Using dataset          : oiv6-mpeg-detection-v1
    Image Scaling          : 100
    Using slice            : 0:2
    Number of samples      : 2
    Progressbar            : False
    Output file            : vtm_out.json
    Print progress         : 1
    
    QUALITY PARAMETER 47
    VTMEncoderDecoder - WARNING - folder /tmp/bitstreams/100/47 exists already
    sample:  1 / 2 tag: 0001eeaf4aed83f9
    sample:  2 / 2 tag: 000a1249af2bc5f0
    
    Done!
    


Instead of generating the bitstreams, the program found them cached on
the disk and just verified them.

Let’s fool around and corrupt one of the bitstreams:

.. code:: ipython3

    echo " " > /tmp/bitstreams/100/47/bin_000a1249af2bc5f0

And run the command again:

.. code:: ipython3

    compressai-vision vtm --y --dataset-name=oiv6-mpeg-detection-v1 \
    --slice=0:2 \
    --scale=100 \
    --progress=1 \
    --qpars=47 \
    --vtm_cache=/tmp/bitstreams \
    --vtm_dir={path_to_vtm_software}/bin \
    --vtm_cfg={path_to_vtm_software}/cfg/encoder_intra_vtm.cfg \
    --output=vtm_out.json


.. code-block:: text

    importing fiftyone
    fiftyone imported
    Reading vtm config from: /home/sampsa/silo/interdigital/VVCSoftware_VTM/cfg/encoder_intra_vtm.cfg
    WARNING: using a dataset slice instead of full dataset: SURE YOU WANT THIS?
    
    VTM bitstream generation
    WARNING: VTM USES CACHE IN /tmp/bitstreams
    Target dir             : /tmp/bitstreams
    Quality points/subdirs : [47]
    Using dataset          : oiv6-mpeg-detection-v1
    Image Scaling          : 100
    Using slice            : 0:2
    Number of samples      : 2
    Progressbar            : False
    Output file            : vtm_out.json
    Print progress         : 1
    
    QUALITY PARAMETER 47
    VTMEncoderDecoder - WARNING - folder /tmp/bitstreams/100/47 exists already
    sample:  1 / 2 tag: 0001eeaf4aed83f9
    VTMEncoderDecoder - CRITICAL - VTM encode failed with Warning: Attempt to decode an empty NAL unit
    
    VTMEncoderDecoder - CRITICAL - VTMDecode failed: will skip image 000a1249af2bc5f0 & remove the bitstream file
    ERROR: Corrupt data for image id=6374fc19f8beb066665b85be, tag=000a1249af2bc5f0, path=/home/sampsa/fiftyone/oiv6-mpeg-detection-v1/data/000a1249af2bc5f0.jpg
    ERROR: Trying to regenerate
    VTMEncoderDecoder - WARNING - creating bitstream /tmp/bitstreams/100/47/bin_000a1249af2bc5f0 with VTMEncode from scratch
    sample:  2 / 2 tag: 000a1249af2bc5f0
    
    Done!
    


You can run the ``vtm`` command parallelized over *both* quality
parameters *and* dataset slices in order to speed things up. In the case
of crashes / data corruption, you can just send the same scripts into
your queue system over and over again if necessary.

Finally, you can run ``detectron2-eval`` for the VTM case like this:

.. code:: ipython3

    compressai-vision detectron2-eval --y --dataset-name=oiv6-mpeg-detection-v1 \
    --slice=0:2 \
    --scale=100 \
    --progress=1 \
    --qpars=47 \
    --vtm \
    --vtm_cache=/tmp/bitstreams \
    --vtm_dir={path_to_vtm_software}/bin \
    --vtm_cfg={path_to_vtm_software}/cfg/encoder_intra_vtm.cfg \
    --output=detectron2_vtm.json \
    --model=COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x.yaml


.. code-block:: text

    importing fiftyone
    fiftyone imported
    WARNING: using a dataset slice instead of full dataset: SURE YOU WANT THIS?
    Reading vtm config from: /home/sampsa/silo/interdigital/VVCSoftware_VTM/cfg/encoder_intra_vtm.cfg
    instantiating Detectron2 predictor 0 : COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x.yaml
    
    Using dataset          : oiv6-mpeg-detection-v1
    Dataset media type     : image
    Dataset tmp clone      : detectron-run-sampsa-oiv6-mpeg-detection-v1-2022-11-16-17-47-58-858646
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
    Using VTM               
    WARNING: VTM USES CACHE IN /tmp/bitstreams
    Quality parameters     : [47]
    Ground truth data field name
                           : detections
    Progressbar            : False
    Print progress         : 1
    Output file            : detectron2_vtm.json
    cloning dataset oiv6-mpeg-detection-v1 to detectron-run-sampsa-oiv6-mpeg-detection-v1-2022-11-16-17-47-58-858646
    VTMEncoderDecoder - WARNING - folder /tmp/bitstreams/100/47 exists already
    /home/sampsa/silo/interdigital/venv_all/lib/python3.8/site-packages/torch/_tensor.py:575: UserWarning: floor_divide is deprecated, and will be removed in a future version of pytorch. It currently rounds toward 0 (like the 'trunc' function NOT 'floor'). This results in incorrect rounding for negative values.
    To keep the current behavior, use torch.div(a, b, rounding_mode='trunc'), or for actual floor division, use torch.div(a, b, rounding_mode='floor'). (Triggered internally at  ../aten/src/ATen/native/BinaryOps.cpp:467.)
      return torch.floor_divide(self, other)
    sample:  1 / 2
    sample:  2 / 2
    Evaluating detections...
    deleting tmp database detectron-run-sampsa-oiv6-mpeg-detection-v1-2022-11-16-17-47-58-858646
    
    Done!
    


