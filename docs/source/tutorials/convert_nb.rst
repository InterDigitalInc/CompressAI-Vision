.. code:: ipython3

    # common libs
    import math, os, io, json, cv2, random, logging
    import numpy as np
    # images
    from PIL import Image
    import matplotlib.pyplot as plt

.. code:: ipython3

    homie=os.path.expanduser("~")
    print("your home path is", homie)
    fodir=os.path.join(homie,'fiftyone')
    print("fiftyone dowloads data to", fodir)
    try:
        os.mkdir(fodir)
    except FileExistsError:
        pass


.. parsed-literal::

    your home path is /home/sampsa
    fiftyone dowloads data to /home/sampsa/fiftyone


.. code:: ipython3

    # fiftyone
    import fiftyone as fo
    import fiftyone.zoo as foz

.. code:: ipython3

    # CompressAI-Vision
    from compressai_vision.conversion import MPEGVCMToOpenImageV6, imageIdFileList

We expect that you have downloaded correct images and segmentation masks
into open-images-v6 folder:

.. code:: ipython3

    dir_=os.path.join(fodir,"open-images-v6")
    print("contents of", dir_,":")
    !tree --filelimit=10 $dir_ | cat


.. parsed-literal::

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


So the downloaded images reside in ``~/fiftyone/open-images-v6/data``
and segmentation masks in ``~/fiftyone/open-images-v6/labels/masks``.

We are not going to use the default OpenImageV6 annotations: MPEG/VCM
working group provides us with custom-format annotation files we need to
convert into OpenImageV6 format. For the detector ground truths, these
are:

::

   detection_validation_5k_bbox.csv           detection bbox annotations
   detection_validation_labels_5k.csv         image-level annotations
   detection_validation_input_5k.lst          list of images used

.. code:: ipython3

    ## TODO: DEFINE YOUR PARTICULAR PATHS
    path_to_mpeg_vcm_files="/home/sampsa/silo/interdigital/siloai-playground/sampsa/mpeg_vcm/data5K"
    path_to_images=os.path.join(fodir,"open-images-v6/validation/data")

    list_file=os.path.join(path_to_mpeg_vcm_files, "detection_validation_input_5k.lst")
    bbox_csv_file=os.path.join(path_to_mpeg_vcm_files, "detection_validation_5k_bbox.csv")
    validation_csv_file=os.path.join(path_to_mpeg_vcm_files, "detection_validation_labels_5k.csv")

    assert(os.path.exists(bbox_csv_file)), "can't find bbox file"
    assert(os.path.exists(validation_csv_file)), "can't find labels file"
    assert(os.path.exists(path_to_images)), "can't find image directory"

Now we convert mpeg_vcm proprietary format annotation into proper
OpenImageV6 format dataset and place it into
``~/fiftyone/mpeg_vcm-detection``

First, remove any previously imported stuff:

.. code:: ipython3

    !rm -rf ~/fiftyone/mpeg_vcm-detection

.. code:: ipython3

    MPEGVCMToOpenImageV6(
        validation_csv_file=validation_csv_file,
        list_file=list_file,
        bbox_csv_file=bbox_csv_file,
        output_directory=os.path.join(fodir,"mpeg_vcm-detection"),
        data_dir=path_to_images
    )

let’s see what we got:

.. code:: ipython3

    !tree --filelimit=10 ~/fiftyone/mpeg_vcm-detection | cat


.. parsed-literal::

    /home/sampsa/fiftyone/mpeg_vcm-detection
    ├── data -> /home/sampsa/fiftyone/open-images-v6/validation/data
    ├── labels
    │   ├── classifications.csv
    │   └── detections.csv
    └── metadata
        ├── attributes.csv
        ├── classes.csv
        └── image_ids.csv

    3 directories, 5 files


We have a new OpenImageV6 formatted data/directory structure with new
annotations, but it uses images from the official OpenImageV6 dataset
(note that link from
``data -> ~/fiftyone/open-images-v6/validation/data``)

The only thing we’re left to do, is to register this OpenImageV6
formatted dataset into fiftyone:

.. code:: ipython3

    # remove the dataset in the case it was already registered in fiftyone
    try:
        fo.delete_dataset("mpeg_vcm-detection")
    except ValueError as e:
        print("could not delete because of", e)

.. code:: ipython3

    dataset_type = fo.types.OpenImagesV6Dataset
    dataset_dir = os.path.join(fodir,"mpeg_vcm-detection")
    dataset = fo.Dataset.from_dir(
        dataset_dir=dataset_dir,
        dataset_type=dataset_type,
        label_types=("detections","classifications"),
        load_hierarchy=False,
        name="mpeg_vcm-detection",
        image_ids=imageIdFileList(list_file)
    )


.. parsed-literal::

     100% |███████████████| 5000/5000 [16.9s elapsed, 0s remaining, 290.3 samples/s]


.. code:: ipython3

    dataset.persistent=True # without this, your dabatase will disappear!

.. code:: ipython3

    ## now, in the future, just do
    dataset = fo.load_dataset("mpeg_vcm-detection")

Finaly, let’s also create a dummy dataset for debugging and testing with
only one sample:

.. code:: ipython3

    try:
        fo.delete_dataset("mpeg_vcm-detection-dummy")
    except ValueError:
        print("no dummmy dataset yet..")
    dummy_dataset=fo.Dataset("mpeg_vcm-detection-dummy")
    for sample in dataset[0:1]:
        dummy_dataset.add_sample(sample)
    dummy_dataset.persistent=True

