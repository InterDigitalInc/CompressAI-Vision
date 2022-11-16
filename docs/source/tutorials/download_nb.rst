In this chapter we use fiftyone to download, inspect and visualize a
subset of OpenImageV6 images

.. code:: ipython3

    # common libs
    import math, os, io, json, cv2, random, logging
    import numpy as np
    # images
    from PIL import Image
    import matplotlib.pyplot as plt

.. code:: ipython3

    # fiftyone
    import fiftyone as fo
    import fiftyone.zoo as foz

.. code:: ipython3

    # CompressAI-Vision
    from compressai_vision.conversion import imageIdFileList

.. code:: ipython3

    homie=os.path.expanduser("~")
    print("your home path is", homie)
    fodir=os.path.join(homie,'fiftyone')
    print("fiftyone dowloads data by default to", fodir)
    try:
        os.mkdir(fodir)
    except FileExistsError:
        pass


.. code-block:: text

    your home path is /home/sampsa
    fiftyone dowloads data by default to /home/sampsa/fiftyone


List all datasets (already) registered to fiftyone

.. code:: ipython3

    fo.list_datasets()




.. parsed-literal::

    ['mpeg-vcm-detection',
     'mpeg-vcm-detection-dummy',
     'mpeg-vcm-segmentation',
     'open-images-v6-validation',
     'quickstart']



Create a list of necessary images

MPEG/VCM working group and mpeg_vcm provides you with files listing the
necessary images for detection and segmentation validation, namely
``detection_validation_input_5k.lst`` and
``segmentation_validation_input_5k.lst``. Let’s combine those two files
into a list:

.. code:: ipython3

    path_to_mpeg_vcm_files="/home/sampsa/silo/interdigital/CompressAI-Vision/compressai_vision/data/mpeg_vcm_data"

.. code:: ipython3

    det_lst=os.path.join(path_to_mpeg_vcm_files,"detection_validation_input_5k.lst") # the images used for detection validation
    seg_lst=os.path.join(path_to_mpeg_vcm_files, "segmentation_validation_input_5k.lst") # the images used for segmentation validation
    assert(os.path.exists(det_lst)), "missing file "+det_lst
    assert(os.path.exists(seg_lst)), "missing file "+seg_lst
    lis=imageIdFileList(det_lst, seg_lst)
    print(len(lis))


.. code-block:: text

    8189


Tell fiftyone to load the correct subset of OpenImageV6 dataset:

.. code:: ipython3

    # https://voxel51.com/docs/fiftyone/user_guide/dataset_zoo/datasets.html#dataset-zoo-open-images-v6
    dataset = foz.load_zoo_dataset(
        "open-images-v6",
        split="validation",
        # label_types=("detections", "classifications", "relationships", "segmentations") # this is the default
        image_ids=lis
    )


.. code-block:: text

    Downloading split 'validation' to '/home/sampsa/fiftyone/open-images-v6/validation' if necessary
    Necessary images already downloaded
    Existing download of split 'validation' is sufficient
    Loading existing dataset 'open-images-v6-validation'. To reload from disk, either delete the existing dataset or provide a custom `dataset_name` to use


.. code:: ipython3

    # take a look at the dataset
    dataset




.. parsed-literal::

    Name:        open-images-v6-validation
    Media type:  image
    Num samples: 8189
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
        relationships:   fiftyone.core.fields.EmbeddedDocumentField(fiftyone.core.labels.Detections)
        segmentations:   fiftyone.core.fields.EmbeddedDocumentField(fiftyone.core.labels.Detections)



.. code:: ipython3

    # make dataset persistent .. next time you import fiftyone it's still available (loaded into the mongodb that's running in the background)
    dataset.persistent=True

.. code:: ipython3

    # next time you need it, load it with:
    dataset = fo.load_dataset("open-images-v6-validation")

.. code:: ipython3

    # peek at first sample
    dataset.first()




.. parsed-literal::

    <Sample: {
        'id': '63371f72ee3965dd2579b526',
        'media_type': 'image',
        'filepath': '/home/sampsa/fiftyone/open-images-v6/validation/data/0001eeaf4aed83f9.jpg',
        'tags': BaseList(['validation']),
        'metadata': None,
        'positive_labels': <Classifications: {
            'classifications': BaseList([
                <Classification: {
                    'id': '63371f72ee3965dd2579b524',
                    'tags': BaseList([]),
                    'label': 'Airplane',
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
                    'id': '63371f72ee3965dd2579b525',
                    'attributes': BaseDict({}),
                    'tags': BaseList([]),
                    'label': 'Airplane',
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
        'relationships': None,
        'segmentations': None,
    }>



Let’s take a look where fiftyone downloaded the files

.. code:: ipython3

    dir_=os.path.join(fodir,"open-images-v6")
    print("contents of", dir_,":")
    !tree --filelimit=10 $dir_ | cat


.. code-block:: text

    contents of /home/sampsa/fiftyone/open-images-v6 :
    /home/sampsa/fiftyone/open-images-v6
    ├── info.json
    └── validation
        ├── data [8189 entries exceeds filelimit, not opening dir]
        ├── labels
        │   ├── classifications.csv
        │   ├── detections.csv
        │   ├── masks [16 entries exceeds filelimit, not opening dir]
        │   ├── relationships.csv
        │   └── segmentations.csv
        └── metadata
            ├── attributes.csv
            ├── classes.csv
            ├── hierarchy.json
            ├── image_ids.csv
            └── segmentation_classes.csv
    
    5 directories, 10 files


if you’d like to remove it, do this:

::

   fo.delete_dataset("open-images-v6-validation")

visualize the dataset with

::

   session = fo.launch_app(dataset)
