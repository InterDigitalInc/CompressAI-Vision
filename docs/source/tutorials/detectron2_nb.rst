3. Detectron2
-------------

-  Going between detectron2 & fiftyone
-  Annexing detectron2 results to fiftyone

In this chapter we look into fiftyone/detectron2 interface, how to add
detectron2 results into a fiftyone dataset and how to evaluate
detectron2 results with fiftyone.

.. code:: ipython3

    # common libs
    import math, os, io, json, cv2, random, logging, datetime
    import numpy as np
    # torch
    import torch
    from torchvision import transforms
    # images
    from PIL import Image
    import matplotlib.pyplot as plt

.. code:: ipython3

    # define a helper function 
    def cv2_imshow(img):
        img2 = img[:,:,::-1]
        plt.figure(figsize=(12, 9))
        plt.axis('off')
        plt.imshow(img2)
        plt.show()

.. code:: ipython3

    ## *** Detectron imports ***
    import detectron2
    from detectron2.utils.logger import setup_logger
    setup_logger()
    
    # import some common detectron2 utilities
    from detectron2 import model_zoo
    from detectron2.engine import DefaultPredictor
    from detectron2.config import get_cfg
    from detectron2.utils.visualizer import Visualizer
    from detectron2.data import MetadataCatalog, DatasetCatalog

.. code:: ipython3

    # CompressAI-Vision
    from compressai_vision.conversion import FO2DetectronDataset # convert fiftyone dataset to Detectron2 dataset
    from compressai_vision.conversion import detectron251 # convert Detectron2 results to fiftyone format
    from compressai_vision.evaluation.fo import annexPredictions # crunch a complete fiftyone dataset through Detectron2 predictor and add the predictions to the fiftyone dataset

.. code:: ipython3

    # fiftyone
    import fiftyone as fo
    import fiftyone.zoo as foz

.. code:: ipython3

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(device)


.. parsed-literal::

    cpu


.. code:: ipython3

    print("torch:", torch.__version__, "/ cuda:", torch.version.cuda, "/ detectron2:", detectron2.__version__)


.. parsed-literal::

    torch: 1.9.1+cu102 / cuda: 10.2 / detectron2: 0.6


Let’s pick up correct Detectron2 model

.. code:: ipython3

    ## MODEL A
    model_name="COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x.yaml"
    ## look here:
    ## https://github.com/facebookresearch/detectron2/blob/main/MODEL_ZOO.md#faster-r-cnn
    
    ## MODEL B
    # model_name="COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"

.. code:: ipython3

    # cfg encapsulates the model architecture & weights, also threshold parameter, metadata, etc.
    cfg = get_cfg()
    cfg.MODEL.DEVICE=device
    # load config from a file:
    cfg.merge_from_file(model_zoo.get_config_file(model_name))
    # DO NOT TOUCH THRESHOLD WHEN DOING EVALUATION:
    # too big a threshold will cut the smallest values & affect the precision(recall) curves & evaluation results
    # the default value is 0.05
    # value of 0.01 saturates the results (they don't change at lower values)
    # cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
    # get weights
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(model_name)
    print("expected input colorspace:", cfg.INPUT.FORMAT)
    print("loaded datasets:", cfg.DATASETS)
    model_dataset=cfg.DATASETS.TRAIN[0]
    print("model was trained with", model_dataset)
    model_meta=MetadataCatalog.get(model_dataset)


.. parsed-literal::

    expected input colorspace: BGR
    loaded datasets: PRECOMPUTED_PROPOSAL_TOPK_TEST: 1000
    PRECOMPUTED_PROPOSAL_TOPK_TRAIN: 2000
    PROPOSAL_FILES_TEST: ()
    PROPOSAL_FILES_TRAIN: ()
    TEST: ('coco_2017_val',)
    TRAIN: ('coco_2017_train',)
    model was trained with coco_2017_train


.. code:: ipython3

    predictor = DefaultPredictor(cfg)

Get a handle to the dataset created in previous notebooks

.. code:: ipython3

    dataset = fo.load_dataset("mpeg-vcm-detection")

.. code:: ipython3

    dataset




.. parsed-literal::

    Name:        mpeg-vcm-detection
    Media type:  image
    Num samples: 5000
    Persistent:  False
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



We can go from fiftyone dataset to Detectron2 dataset:

.. code:: ipython3

    detectron_dataset=FO2DetectronDataset(fo_dataset=dataset, model_catids=model_meta.thing_classes)

Pick a sample:

.. code:: ipython3

    d=detectron_dataset[3]

We can visualize that sample also with Detectron2 library tools
(although we’d prefer fiftyone with ``fo.launch_app(dataset)``):

.. code:: ipython3

    # visualize with Detectron2 tools only
    img = cv2.imread(d["file_name"])
    visualizer = Visualizer(img[:, :, ::-1], metadata=model_meta, scale=0.5)
    out = visualizer.draw_dataset_dict(d)
    cv2_imshow(out.get_image()[:, :, ::-1])



.. image:: detectron2_nb_files/detectron2_nb_21_0.png


Let’s try the Detectron2 predictor:

.. code:: ipython3

    res=predictor(img)


.. parsed-literal::

    /home/sampsa/silo/interdigital/venv_all/lib/python3.8/site-packages/torch/_tensor.py:575: UserWarning: floor_divide is deprecated, and will be removed in a future version of pytorch. It currently rounds toward 0 (like the 'trunc' function NOT 'floor'). This results in incorrect rounding for negative values.
    To keep the current behavior, use torch.div(a, b, rounding_mode='trunc'), or for actual floor division, use torch.div(a, b, rounding_mode='floor'). (Triggered internally at  ../aten/src/ATen/native/BinaryOps.cpp:467.)
      return torch.floor_divide(self, other)


We can convert from Detectron2 format to fiftyone detection objects:

.. code:: ipython3

    dets=detectron251(res, model_catids=model_meta.thing_classes) # process involves going from class indexes (ints) to class labels (strings)

.. code:: ipython3

    dets




.. parsed-literal::

    <Detections: {
        'detections': BaseList([
            <Detection: {
                'id': '6346dd58f9f8269320fa1e9b',
                'attributes': BaseDict({}),
                'tags': BaseList([]),
                'label': 'person',
                'bounding_box': BaseList([
                    0.009324110113084316,
                    0.07061359169804884,
                    0.636555933393538,
                    0.9101487120042683,
                ]),
                'mask': None,
                'confidence': 0.9894514679908752,
                'index': None,
            }>,
            <Detection: {
                'id': '6346dd58f9f8269320fa1e9c',
                'attributes': BaseDict({}),
                'tags': BaseList([]),
                'label': 'person',
                'bounding_box': BaseList([
                    0.7662928700447083,
                    0.8120538199233528,
                    0.13444077968597412,
                    0.18774914350665983,
                ]),
                'mask': None,
                'confidence': 0.9372856616973877,
                'index': None,
            }>,
            <Detection: {
                'id': '6346dd58f9f8269320fa1e9d',
                'attributes': BaseDict({}),
                'tags': BaseList([]),
                'label': 'person',
                'bounding_box': BaseList([
                    0.6052085757255554,
                    0.8155682288382724,
                    0.20704376697540283,
                    0.18033525772383355,
                ]),
                'mask': None,
                'confidence': 0.9026966094970703,
                'index': None,
            }>,
            <Detection: {
                'id': '6346dd58f9f8269320fa1e9e',
                'attributes': BaseDict({}),
                'tags': BaseList([]),
                'label': 'motorcycle',
                'bounding_box': BaseList([
                    0.485452800989151,
                    0.016184425594527062,
                    0.40658149123191833,
                    0.8690680952420902,
                ]),
                'mask': None,
                'confidence': 0.7087109684944153,
                'index': None,
            }>,
            <Detection: {
                'id': '6346dd58f9f8269320fa1e9f',
                'attributes': BaseDict({}),
                'tags': BaseList([]),
                'label': 'person',
                'bounding_box': BaseList([
                    0.014408787712454796,
                    0.5796498013625079,
                    0.6207742486149073,
                    0.40208690975232503,
                ]),
                'mask': None,
                'confidence': 0.6055470108985901,
                'index': None,
            }>,
            <Detection: {
                'id': '6346dd58f9f8269320fa1ea0',
                'attributes': BaseDict({}),
                'tags': BaseList([]),
                'label': 'person',
                'bounding_box': BaseList([
                    0.2356555312871933,
                    0.4367165102483646,
                    0.26028211414813995,
                    0.4146875138541338,
                ]),
                'mask': None,
                'confidence': 0.3441943824291229,
                'index': None,
            }>,
            <Detection: {
                'id': '6346dd58f9f8269320fa1ea1',
                'attributes': BaseDict({}),
                'tags': BaseList([]),
                'label': 'bicycle',
                'bounding_box': BaseList([
                    0.4535953998565674,
                    0.006116752215622352,
                    0.4246022701263428,
                    0.9400808730011179,
                ]),
                'mask': None,
                'confidence': 0.291936993598938,
                'index': None,
            }>,
            <Detection: {
                'id': '6346dd58f9f8269320fa1ea2',
                'attributes': BaseDict({}),
                'tags': BaseList([]),
                'label': 'person',
                'bounding_box': BaseList([
                    0.7321376204490662,
                    0.8158130621699637,
                    0.10273182392120361,
                    0.1786184226729981,
                ]),
                'mask': None,
                'confidence': 0.251709520816803,
                'index': None,
            }>,
            <Detection: {
                'id': '6346dd58f9f8269320fa1ea3',
                'attributes': BaseDict({}),
                'tags': BaseList([]),
                'label': 'cell phone',
                'bounding_box': BaseList([
                    0.5621044039726257,
                    0.9012914515684111,
                    0.039388060569763184,
                    0.04222701506837169,
                ]),
                'mask': None,
                'confidence': 0.2146982103586197,
                'index': None,
            }>,
            <Detection: {
                'id': '6346dd58f9f8269320fa1ea4',
                'attributes': BaseDict({}),
                'tags': BaseList([]),
                'label': 'person',
                'bounding_box': BaseList([
                    0.29376694560050964,
                    0.46854041774215793,
                    0.1675676703453064,
                    0.29636893927825503,
                ]),
                'mask': None,
                'confidence': 0.15639427304267883,
                'index': None,
            }>,
            <Detection: {
                'id': '6346dd58f9f8269320fa1ea5',
                'attributes': BaseDict({}),
                'tags': BaseList([]),
                'label': 'person',
                'bounding_box': BaseList([
                    0.004228693433105946,
                    0.0767963404613365,
                    0.37698595505207777,
                    0.630363829950667,
                ]),
                'mask': None,
                'confidence': 0.11882766336202621,
                'index': None,
            }>,
            <Detection: {
                'id': '6346dd58f9f8269320fa1ea6',
                'attributes': BaseDict({}),
                'tags': BaseList([]),
                'label': 'person',
                'bounding_box': BaseList([
                    0.6203261017799377,
                    0.865912842720484,
                    0.09154510498046875,
                    0.13247769176584173,
                ]),
                'mask': None,
                'confidence': 0.09884662926197052,
                'index': None,
            }>,
            <Detection: {
                'id': '6346dd58f9f8269320fa1ea7',
                'attributes': BaseDict({}),
                'tags': BaseList([]),
                'label': 'person',
                'bounding_box': BaseList([
                    0.08518574386835098,
                    0.3548633628354548,
                    0.4390302523970604,
                    0.5531069357488375,
                ]),
                'mask': None,
                'confidence': 0.0946485623717308,
                'index': None,
            }>,
            <Detection: {
                'id': '6346dd58f9f8269320fa1ea8',
                'attributes': BaseDict({}),
                'tags': BaseList([]),
                'label': 'motorcycle',
                'bounding_box': BaseList([
                    0.1718100905418396,
                    0.0204875556397408,
                    0.6795393824577332,
                    0.8203254449562948,
                ]),
                'mask': None,
                'confidence': 0.08924849331378937,
                'index': None,
            }>,
            <Detection: {
                'id': '6346dd58f9f8269320fa1ea9',
                'attributes': BaseDict({}),
                'tags': BaseList([]),
                'label': 'truck',
                'bounding_box': BaseList([
                    0.23365649580955505,
                    0.0,
                    0.7420808374881744,
                    0.9733538826056116,
                ]),
                'mask': None,
                'confidence': 0.07246677577495575,
                'index': None,
            }>,
            <Detection: {
                'id': '6346dd58f9f8269320fa1eaa',
                'attributes': BaseDict({}),
                'tags': BaseList([]),
                'label': 'baseball bat',
                'bounding_box': BaseList([
                    0.8032967448234558,
                    0.2696970185596134,
                    0.07617664337158203,
                    0.2461811272523743,
                ]),
                'mask': None,
                'confidence': 0.06142522394657135,
                'index': None,
            }>,
        ]),
    }>



Let’s run each image in a fiftyone dataset through the predictor.
Results from the predictor will be annexed to the same fiftyone dataset.
We use the dummy single-sample dataset ``mpeg-vcm-detection-dummy``
created in previous chapters for testing:

.. code:: ipython3

    dataset = fo.load_dataset("mpeg-vcm-detection-dummy")

Detectron prediction results are saved during the run into the fiftyone
(mongodb) database. Let’s define a unique name for the sample field
where the detectron results will be saved:

.. code:: ipython3

    predictor_field='detectron-predictions'

.. code:: ipython3

    annexPredictions(predictor=predictor, fo_dataset=dataset, predictor_field=predictor_field)


.. parsed-literal::

    sample:  1 / 1


After that one, the dataset looks slightly different. Take note that an
extra field ``detectron-predictions`` has appeared into the dataset:

.. code:: ipython3

    print(dataset)


.. parsed-literal::

    Name:        mpeg-vcm-detection-dummy
    Media type:  image
    Num samples: 1
    Persistent:  True
    Tags:        []
    Sample fields:
        id:                    fiftyone.core.fields.ObjectIdField
        filepath:              fiftyone.core.fields.StringField
        tags:                  fiftyone.core.fields.ListField(fiftyone.core.fields.StringField)
        metadata:              fiftyone.core.fields.EmbeddedDocumentField(fiftyone.core.metadata.ImageMetadata)
        positive_labels:       fiftyone.core.fields.EmbeddedDocumentField(fiftyone.core.labels.Classifications)
        negative_labels:       fiftyone.core.fields.EmbeddedDocumentField(fiftyone.core.labels.Classifications)
        detections:            fiftyone.core.fields.EmbeddedDocumentField(fiftyone.core.labels.Detections)
        open_images_id:        fiftyone.core.fields.StringField
        detectron-predictions: fiftyone.core.fields.EmbeddedDocumentField(fiftyone.core.labels.Detections)


Let’s peek at the first sample:

.. code:: ipython3

    sample=dataset.first()

.. code:: ipython3

    print(sample)


.. parsed-literal::

    <Sample: {
        'id': '6346d4246b7dd87eda0f2b04',
        'media_type': 'image',
        'filepath': '/home/sampsa/fiftyone/mpeg-vcm-detection/data/0001eeaf4aed83f9.jpg',
        'tags': BaseList([]),
        'metadata': None,
        'positive_labels': <Classifications: {
            'classifications': BaseList([
                <Classification: {
                    'id': '6346d4246b7dd87eda0f2b02',
                    'tags': BaseList([]),
                    'label': 'airplane',
                    'confidence': 1.0,
                    'logits': None,
                }>,
            ]),
            'logits': None,
        }>,
        'negative_labels': <Classifications: {'classifications': BaseList([]), 'logits': None}>,
        'detections': <Detections: {
            'detections': BaseList([
                <Detection: {
                    'id': '6346d4246b7dd87eda0f2b03',
                    'attributes': BaseDict({}),
                    'tags': BaseList([]),
                    'label': 'airplane',
                    'bounding_box': BaseList([
                        0.022673031,
                        0.07103825,
                        0.9415274690000001,
                        0.72950822,
                    ]),
                    'mask': None,
                    'confidence': None,
                    'index': None,
                    'IsOccluded': False,
                    'IsTruncated': False,
                    'IsGroupOf': False,
                    'IsDepiction': False,
                    'IsInside': False,
                }>,
            ]),
        }>,
        'open_images_id': '0001eeaf4aed83f9',
        'detectron-predictions': <Detections: {
            'detections': BaseList([
                <Detection: {
                    'id': '6346df3ef9f8269320fa1efd',
                    'attributes': BaseDict({}),
                    'tags': BaseList([]),
                    'label': 'airplane',
                    'bounding_box': BaseList([
                        0.12768225371837616,
                        0.027486978227926846,
                        0.8119979053735733,
                        0.76541841536827,
                    ]),
                    'mask': None,
                    'confidence': 0.9963523149490356,
                    'index': None,
                }>,
                <Detection: {
                    'id': '6346df3ef9f8269320fa1efe',
                    'attributes': BaseDict({}),
                    'tags': BaseList([]),
                    'label': 'truck',
                    'bounding_box': BaseList([
                        0.9190815091133118,
                        0.6016124750943792,
                        0.0368688702583313,
                        0.07163922952058864,
                    ]),
                    'mask': None,
                    'confidence': 0.9494412541389465,
                    'index': None,
                }>,
                <Detection: {
                    'id': '6346df3ef9f8269320fa1eff',
                    'attributes': BaseDict({}),
                    'tags': BaseList([]),
                    'label': 'truck',
                    'bounding_box': BaseList([
                        0.8702287077903748,
                        0.6155941683707354,
                        0.050144076347351074,
                        0.055400259542785234,
                    ]),
                    'mask': None,
                    'confidence': 0.9203835725784302,
                    'index': None,
                }>,
                <Detection: {
                    'id': '6346df3ef9f8269320fa1f00',
                    'attributes': BaseDict({}),
                    'tags': BaseList([]),
                    'label': 'airplane',
                    'bounding_box': BaseList([
                        0.11006083339452744,
                        0.3412696787174916,
                        0.19038165360689163,
                        0.19853596185944491,
                    ]),
                    'mask': None,
                    'confidence': 0.7442840933799744,
                    'index': None,
                }>,
                <Detection: {
                    'id': '6346df3ef9f8269320fa1f01',
                    'attributes': BaseDict({}),
                    'tags': BaseList([]),
                    'label': 'truck',
                    'bounding_box': BaseList([
                        0.7966880202293396,
                        0.6970266730460011,
                        0.08078855276107788,
                        0.07823143602750979,
                    ]),
                    'mask': None,
                    'confidence': 0.6773070096969604,
                    'index': None,
                }>,
                <Detection: {
                    'id': '6346df3ef9f8269320fa1f02',
                    'attributes': BaseDict({}),
                    'tags': BaseList([]),
                    'label': 'truck',
                    'bounding_box': BaseList([
                        0.19424161314964294,
                        0.5731003865832984,
                        0.03333866596221924,
                        0.04234303724045722,
                    ]),
                    'mask': None,
                    'confidence': 0.49123454093933105,
                    'index': None,
                }>,
                <Detection: {
                    'id': '6346df3ef9f8269320fa1f03',
                    'attributes': BaseDict({}),
                    'tags': BaseList([]),
                    'label': 'car',
                    'bounding_box': BaseList([
                        0.8586535453796387,
                        0.6244919198738115,
                        0.013080716133117676,
                        0.040624153427362975,
                    ]),
                    'mask': None,
                    'confidence': 0.49095118045806885,
                    'index': None,
                }>,
                <Detection: {
                    'id': '6346df3ef9f8269320fa1f04',
                    'attributes': BaseDict({}),
                    'tags': BaseList([]),
                    'label': 'airplane',
                    'bounding_box': BaseList([
                        0.11007525771856308,
                        0.34265331293912543,
                        0.10588719695806503,
                        0.17782678113421072,
                    ]),
                    'mask': None,
                    'confidence': 0.4769124388694763,
                    'index': None,
                }>,
                <Detection: {
                    'id': '6346df3ef9f8269320fa1f05',
                    'attributes': BaseDict({}),
                    'tags': BaseList([]),
                    'label': 'person',
                    'bounding_box': BaseList([
                        0.8444294929504395,
                        0.6848512517259159,
                        0.006960868835449219,
                        0.031434061276566,
                    ]),
                    'mask': None,
                    'confidence': 0.39503416419029236,
                    'index': None,
                }>,
                <Detection: {
                    'id': '6346df3ef9f8269320fa1f06',
                    'attributes': BaseDict({}),
                    'tags': BaseList([]),
                    'label': 'car',
                    'bounding_box': BaseList([
                        0.11215677112340927,
                        0.5923443480626048,
                        0.04038272053003311,
                        0.033441189418169745,
                    ]),
                    'mask': None,
                    'confidence': 0.38696712255477905,
                    'index': None,
                }>,
                <Detection: {
                    'id': '6346df3ef9f8269320fa1f07',
                    'attributes': BaseDict({}),
                    'tags': BaseList([]),
                    'label': 'airplane',
                    'bounding_box': BaseList([
                        0.10338009148836136,
                        0.5782626363255033,
                        0.05053221434354782,
                        0.05472764392827181,
                    ]),
                    'mask': None,
                    'confidence': 0.36884164810180664,
                    'index': None,
                }>,
                <Detection: {
                    'id': '6346df3ef9f8269320fa1f08',
                    'attributes': BaseDict({}),
                    'tags': BaseList([]),
                    'label': 'airplane',
                    'bounding_box': BaseList([
                        0.0255854744464159,
                        0.5004235935424531,
                        0.2138556968420744,
                        0.13064667362494756,
                    ]),
                    'mask': None,
                    'confidence': 0.3492622375488281,
                    'index': None,
                }>,
                <Detection: {
                    'id': '6346df3ef9f8269320fa1f09',
                    'attributes': BaseDict({}),
                    'tags': BaseList([]),
                    'label': 'airplane',
                    'bounding_box': BaseList([
                        0.14727365970611572,
                        0.34949549732592283,
                        0.054032012820243835,
                        0.08753325528479795,
                    ]),
                    'mask': None,
                    'confidence': 0.33867529034614563,
                    'index': None,
                }>,
                <Detection: {
                    'id': '6346df3ef9f8269320fa1f0a',
                    'attributes': BaseDict({}),
                    'tags': BaseList([]),
                    'label': 'car',
                    'bounding_box': BaseList([
                        0.19401228427886963,
                        0.5745964989032788,
                        0.033640727400779724,
                        0.04079626710622903,
                    ]),
                    'mask': None,
                    'confidence': 0.30962127447128296,
                    'index': None,
                }>,
                <Detection: {
                    'id': '6346df3ef9f8269320fa1f0b',
                    'attributes': BaseDict({}),
                    'tags': BaseList([]),
                    'label': 'truck',
                    'bounding_box': BaseList([
                        0.9722632169723511,
                        0.6104661228939458,
                        0.02353382110595703,
                        0.032473229188513704,
                    ]),
                    'mask': None,
                    'confidence': 0.3069209158420563,
                    'index': None,
                }>,
                <Detection: {
                    'id': '6346df3ef9f8269320fa1f0c',
                    'attributes': BaseDict({}),
                    'tags': BaseList([]),
                    'label': 'airplane',
                    'bounding_box': BaseList([
                        0.14766937494277954,
                        0.35212325196404853,
                        0.07240810990333557,
                        0.1256532605062395,
                    ]),
                    'mask': None,
                    'confidence': 0.2942286729812622,
                    'index': None,
                }>,
                <Detection: {
                    'id': '6346df3ef9f8269320fa1f0d',
                    'attributes': BaseDict({}),
                    'tags': BaseList([]),
                    'label': 'truck',
                    'bounding_box': BaseList([
                        0.8577467799186707,
                        0.6242693531966583,
                        0.013565301895141602,
                        0.042288283106998045,
                    ]),
                    'mask': None,
                    'confidence': 0.2310165911912918,
                    'index': None,
                }>,
                <Detection: {
                    'id': '6346df3ef9f8269320fa1f0e',
                    'attributes': BaseDict({}),
                    'tags': BaseList([]),
                    'label': 'person',
                    'bounding_box': BaseList([
                        0.8507804274559021,
                        0.6853243083228467,
                        0.007759749889373779,
                        0.027981273813269016,
                    ]),
                    'mask': None,
                    'confidence': 0.2067108452320099,
                    'index': None,
                }>,
                <Detection: {
                    'id': '6346df3ef9f8269320fa1f0f',
                    'attributes': BaseDict({}),
                    'tags': BaseList([]),
                    'label': 'truck',
                    'bounding_box': BaseList([
                        0.10813499987125397,
                        0.5906133801078369,
                        0.04447351396083832,
                        0.03619398420022371,
                    ]),
                    'mask': None,
                    'confidence': 0.20025715231895447,
                    'index': None,
                }>,
                <Detection: {
                    'id': '6346df3ef9f8269320fa1f10',
                    'attributes': BaseDict({}),
                    'tags': BaseList([]),
                    'label': 'truck',
                    'bounding_box': BaseList([
                        0.1583663523197174,
                        0.5795836309991961,
                        0.015417307615280151,
                        0.035827704990736856,
                    ]),
                    'mask': None,
                    'confidence': 0.1864553838968277,
                    'index': None,
                }>,
                <Detection: {
                    'id': '6346df3ef9f8269320fa1f11',
                    'attributes': BaseDict({}),
                    'tags': BaseList([]),
                    'label': 'airplane',
                    'bounding_box': BaseList([
                        0.15734650194644928,
                        0.34015182734069144,
                        0.09979219734668732,
                        0.17215952350522581,
                    ]),
                    'mask': None,
                    'confidence': 0.16270428895950317,
                    'index': None,
                }>,
                <Detection: {
                    'id': '6346df3ef9f8269320fa1f12',
                    'attributes': BaseDict({}),
                    'tags': BaseList([]),
                    'label': 'car',
                    'bounding_box': BaseList([
                        0.9724457263946533,
                        0.6110104554451552,
                        0.022782206535339355,
                        0.03191920201517058,
                    ]),
                    'mask': None,
                    'confidence': 0.14726606011390686,
                    'index': None,
                }>,
                <Detection: {
                    'id': '6346df3ef9f8269320fa1f13',
                    'attributes': BaseDict({}),
                    'tags': BaseList([]),
                    'label': 'airplane',
                    'bounding_box': BaseList([
                        0.9310635328292847,
                        0.46706545699629476,
                        0.06893646717071533,
                        0.07955963339581586,
                    ]),
                    'mask': None,
                    'confidence': 0.14320485293865204,
                    'index': None,
                }>,
                <Detection: {
                    'id': '6346df3ef9f8269320fa1f14',
                    'attributes': BaseDict({}),
                    'tags': BaseList([]),
                    'label': 'airplane',
                    'bounding_box': BaseList([
                        0.10666287690401077,
                        0.2859448597201832,
                        0.2402758225798607,
                        0.3414595431129404,
                    ]),
                    'mask': None,
                    'confidence': 0.1379343718290329,
                    'index': None,
                }>,
                <Detection: {
                    'id': '6346df3ef9f8269320fa1f15',
                    'attributes': BaseDict({}),
                    'tags': BaseList([]),
                    'label': 'truck',
                    'bounding_box': BaseList([
                        0.8471341729164124,
                        0.616852284544533,
                        0.05158036947250366,
                        0.054390312041212245,
                    ]),
                    'mask': None,
                    'confidence': 0.13730646669864655,
                    'index': None,
                }>,
                <Detection: {
                    'id': '6346df3ef9f8269320fa1f16',
                    'attributes': BaseDict({}),
                    'tags': BaseList([]),
                    'label': 'car',
                    'bounding_box': BaseList([
                        0.9190192222595215,
                        0.6098847186538731,
                        0.0384480357170105,
                        0.06279070212003636,
                    ]),
                    'mask': None,
                    'confidence': 0.13430571556091309,
                    'index': None,
                }>,
                <Detection: {
                    'id': '6346df3ef9f8269320fa1f17',
                    'attributes': BaseDict({}),
                    'tags': BaseList([]),
                    'label': 'airplane',
                    'bounding_box': BaseList([
                        0.028436832129955292,
                        0.3380399699712493,
                        0.22196123749017715,
                        0.25432984354245314,
                    ]),
                    'mask': None,
                    'confidence': 0.12494388967752457,
                    'index': None,
                }>,
                <Detection: {
                    'id': '6346df3ef9f8269320fa1f18',
                    'attributes': BaseDict({}),
                    'tags': BaseList([]),
                    'label': 'person',
                    'bounding_box': BaseList([
                        0.8357962965965271,
                        0.6915914719003425,
                        0.00689089298248291,
                        0.029422905087737695,
                    ]),
                    'mask': None,
                    'confidence': 0.12231158465147018,
                    'index': None,
                }>,
                <Detection: {
                    'id': '6346df3ef9f8269320fa1f19',
                    'attributes': BaseDict({}),
                    'tags': BaseList([]),
                    'label': 'truck',
                    'bounding_box': BaseList([
                        0.8622568249702454,
                        0.6034765734235179,
                        0.09968775510787964,
                        0.06998854942236438,
                    ]),
                    'mask': None,
                    'confidence': 0.1009499728679657,
                    'index': None,
                }>,
                <Detection: {
                    'id': '6346df3ef9f8269320fa1f1a',
                    'attributes': BaseDict({}),
                    'tags': BaseList([]),
                    'label': 'airplane',
                    'bounding_box': BaseList([
                        0.11483647674322128,
                        0.5900956736315017,
                        0.0387362465262413,
                        0.033792312246574384,
                    ]),
                    'mask': None,
                    'confidence': 0.09404828399419785,
                    'index': None,
                }>,
                <Detection: {
                    'id': '6346df3ef9f8269320fa1f1b',
                    'attributes': BaseDict({}),
                    'tags': BaseList([]),
                    'label': 'person',
                    'bounding_box': BaseList([
                        0.8381629586219788,
                        0.690306183475776,
                        0.007335245609283447,
                        0.029042561848958332,
                    ]),
                    'mask': None,
                    'confidence': 0.09348491579294205,
                    'index': None,
                }>,
                <Detection: {
                    'id': '6346df3ef9f8269320fa1f1c',
                    'attributes': BaseDict({}),
                    'tags': BaseList([]),
                    'label': 'person',
                    'bounding_box': BaseList([
                        0.8437240719795227,
                        0.6959601280673239,
                        0.005839407444000244,
                        0.021766270033731824,
                    ]),
                    'mask': None,
                    'confidence': 0.09292487055063248,
                    'index': None,
                }>,
                <Detection: {
                    'id': '6346df3ef9f8269320fa1f1d',
                    'attributes': BaseDict({}),
                    'tags': BaseList([]),
                    'label': 'airplane',
                    'bounding_box': BaseList([
                        0.19413751363754272,
                        0.3352931148520519,
                        0.06308001279830933,
                        0.14197770701158766,
                    ]),
                    'mask': None,
                    'confidence': 0.08922168612480164,
                    'index': None,
                }>,
                <Detection: {
                    'id': '6346df3ef9f8269320fa1f1e',
                    'attributes': BaseDict({}),
                    'tags': BaseList([]),
                    'label': 'person',
                    'bounding_box': BaseList([
                        0.8467901349067688,
                        0.6858217380190855,
                        0.00686413049697876,
                        0.03015614622657998,
                    ]),
                    'mask': None,
                    'confidence': 0.07978574186563492,
                    'index': None,
                }>,
                <Detection: {
                    'id': '6346df3ef9f8269320fa1f1f',
                    'attributes': BaseDict({}),
                    'tags': BaseList([]),
                    'label': 'truck',
                    'bounding_box': BaseList([
                        0.09427589178085327,
                        0.5730622225426454,
                        0.127224862575531,
                        0.046534322785584455,
                    ]),
                    'mask': None,
                    'confidence': 0.06778866797685623,
                    'index': None,
                }>,
                <Detection: {
                    'id': '6346df3ef9f8269320fa1f20',
                    'attributes': BaseDict({}),
                    'tags': BaseList([]),
                    'label': 'person',
                    'bounding_box': BaseList([
                        0.8454108834266663,
                        0.6975023239784326,
                        0.006599128246307373,
                        0.019551595052083332,
                    ]),
                    'mask': None,
                    'confidence': 0.06577832251787186,
                    'index': None,
                }>,
                <Detection: {
                    'id': '6346df3ef9f8269320fa1f21',
                    'attributes': BaseDict({}),
                    'tags': BaseList([]),
                    'label': 'person',
                    'bounding_box': BaseList([
                        0.8358362317085266,
                        0.6818531556950853,
                        0.01932704448699951,
                        0.03676426010643876,
                    ]),
                    'mask': None,
                    'confidence': 0.05755734071135521,
                    'index': None,
                }>,
                <Detection: {
                    'id': '6346df3ef9f8269320fa1f22',
                    'attributes': BaseDict({}),
                    'tags': BaseList([]),
                    'label': 'person',
                    'bounding_box': BaseList([
                        0.13454458117485046,
                        0.6153508470095778,
                        0.009050920605659485,
                        0.03343163134000979,
                    ]),
                    'mask': None,
                    'confidence': 0.05496569350361824,
                    'index': None,
                }>,
                <Detection: {
                    'id': '6346df3ef9f8269320fa1f23',
                    'attributes': BaseDict({}),
                    'tags': BaseList([]),
                    'label': 'person',
                    'bounding_box': BaseList([
                        0.10967192053794861,
                        0.5790209332835221,
                        0.009264662861824036,
                        0.036160940558585014,
                    ]),
                    'mask': None,
                    'confidence': 0.05112108960747719,
                    'index': None,
                }>,
            ]),
        }>,
    }>


Each sample in the dataset contains “detections” (ground truths) and
“detectron-predictions” (predicted values). Now we can run the
OpenImageV6 evaluation protocol on the dataset which uses the ground
truth and the predictor results:

.. code:: ipython3

    results = dataset.evaluate_detections(
        predictor_field,
        gt_field="detections",
        method="open-images",
        pos_label_field="positive_labels",
        neg_label_field="negative_labels",
        expand_pred_hierarchy=False,
        expand_gt_hierarchy=False
    )


.. parsed-literal::

    Evaluating detections...
     100% |█████████████████████| 1/1 [31.8ms elapsed, 0s remaining, 31.4 samples/s] 


After the evaluation we can should remove the detectron results from the
database:

.. code:: ipython3

    dataset.delete_sample_fields(predictor_field)

OpenImageV6 evaluation protocol mAP:

.. code:: ipython3

    results.mAP()




.. parsed-literal::

    1.0



Per class mAP:

.. code:: ipython3

    classes = dataset.distinct(
        "detections.detections.label"
    )
    for class_ in classes:
        print(class_, results.mAP([class_]))


.. parsed-literal::

    airplane 1.0


In practice (and what the CLI program does) it is a better idea to
create a copy of the complete dataset into a temporary dataset for
appending detection results (especially if you are sharing datasets in
your grid/cluster) and after getting the mAP results to remove the
temporary dataset. On how to do this, please refer to the fiftyone
documentation.

