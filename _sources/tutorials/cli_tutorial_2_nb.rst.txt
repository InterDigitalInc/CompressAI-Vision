In this chapter you will learn:

-  registering and deregistering datasets into fiftyone

In the previous chapter we downloaded & registered the dataset
“quickstart” from the fiftyone model zoo:

.. code:: bash

    compressai-vision list


.. code-block:: text

    importing fiftyone
    fiftyone imported
    
    datasets currently registered into fiftyone
    name, length, first sample path
    mpeg-vcm-detection, 5000, /home/sampsa/fiftyone/mpeg-vcm-detection/data
    mpeg-vcm-detection-dummy, 1, /home/sampsa/fiftyone/mpeg-vcm-detection/data
    mpeg-vcm-segmentation, 5000, /home/sampsa/fiftyone/mpeg-vcm-segmentation/data
    open-images-v6-validation, 8189, /home/sampsa/fiftyone/open-images-v6/validation/data
    quickstart, 200, /home/sampsa/fiftyone/quickstart/data


All the metadata, ground truth bboxes, etc. reside now in the
fiftyone/mongodb database at dataset “quickstart”. That data was read
into the database originally from directory ``~/fiftyone/quickstart``.
Let’s see what’s in there:

.. code:: bash

    ls ~/fiftyone/quickstart


.. code-block:: text

    data  info.json  metadata.json	samples.json


Exactly. Note directory ``data``. That is where the sample images are
(they are *not* in the database, but just on the disk as image files).
Let’s take a look at that:

.. code:: bash

    ls ~/fiftyone/quickstart/data


.. code-block:: text

    000002.jpg  000889.jpg	001851.jpg  002598.jpg	003754.jpg  004416.jpg
    000008.jpg  000890.jpg	001867.jpg  002640.jpg	003769.jpg  004431.jpg
    000020.jpg  000939.jpg	001888.jpg  002645.jpg	003805.jpg  004510.jpg
    000031.jpg  000957.jpg	001934.jpg  002660.jpg	003870.jpg  004514.jpg
    000035.jpg  000998.jpg	001949.jpg  002671.jpg	003871.jpg  004517.jpg
    000058.jpg  001047.jpg	001951.jpg  002748.jpg	003880.jpg  004525.jpg
    000083.jpg  001057.jpg	001983.jpg  002799.jpg	003888.jpg  004534.jpg
    000089.jpg  001073.jpg	002015.jpg  002823.jpg	003911.jpg  004535.jpg
    000145.jpg  001078.jpg	002022.jpg  002869.jpg	003964.jpg  004546.jpg
    000164.jpg  001118.jpg	002063.jpg  002905.jpg	003969.jpg  004548.jpg
    000191.jpg  001133.jpg	002070.jpg  002906.jpg	003978.jpg  004557.jpg
    000192.jpg  001147.jpg	002086.jpg  002939.jpg	004039.jpg  004585.jpg
    000400.jpg  001154.jpg	002121.jpg  002953.jpg	004066.jpg  004590.jpg
    000436.jpg  001191.jpg	002129.jpg  003084.jpg	004082.jpg  004610.jpg
    000452.jpg  001227.jpg	002143.jpg  003087.jpg	004095.jpg  004627.jpg
    000496.jpg  001289.jpg	002184.jpg  003107.jpg	004096.jpg  004651.jpg
    000510.jpg  001312.jpg	002186.jpg  003132.jpg	004126.jpg  004656.jpg
    000557.jpg  001348.jpg	002233.jpg  003148.jpg	004131.jpg  004702.jpg
    000575.jpg  001394.jpg	002284.jpg  003154.jpg	004170.jpg  004713.jpg
    000591.jpg  001429.jpg	002334.jpg  003254.jpg	004172.jpg  004743.jpg
    000594.jpg  001430.jpg	002353.jpg  003316.jpg	004180.jpg  004755.jpg
    000600.jpg  001586.jpg	002431.jpg  003344.jpg	004222.jpg  004775.jpg
    000641.jpg  001587.jpg	002439.jpg  003391.jpg	004253.jpg  004781.jpg
    000643.jpg  001599.jpg	002450.jpg  003420.jpg	004255.jpg  004831.jpg
    000648.jpg  001614.jpg	002462.jpg  003486.jpg	004263.jpg  004852.jpg
    000665.jpg  001624.jpg	002468.jpg  003502.jpg	004284.jpg  004939.jpg
    000696.jpg  001631.jpg	002489.jpg  003541.jpg	004292.jpg  004941.jpg
    000772.jpg  001634.jpg	002497.jpg  003614.jpg	004304.jpg  004965.jpg
    000773.jpg  001645.jpg	002514.jpg  003659.jpg	004315.jpg  004978.jpg
    000781.jpg  001685.jpg	002538.jpg  003662.jpg	004316.jpg  004981.jpg
    000793.jpg  001698.jpg	002553.jpg  003665.jpg	004329.jpg
    000807.jpg  001741.jpg	002586.jpg  003667.jpg	004330.jpg
    000868.jpg  001744.jpg	002592.jpg  003713.jpg	004341.jpg
    000880.jpg  001763.jpg	002597.jpg  003715.jpg	004371.jpg


The fiftyone dataset “quickstart” has only the paths to these files.

Next suppose you have a dataset already on your disk (say, on the
ImageDir format, COCO format, whatever) and you wish to register it into
fiftyone.

In order to demo that, let’s create a copy of ``~/fiftyone/quickstart``:

.. code:: bash

    cp -r ~/fiftyone/quickstart /tmp/my_data_set

Let’s imagine ``/tmp/my_data_set`` is that custom dataset of yours you
had already lying around.

We register it to fiftyone with:

.. code:: bash

    compressai-vision register --y \
    --dataset-name=my_dataset \
    --dir=/tmp/my_data_set \
    --type=FiftyOneDataset


.. code-block:: text

    importing fiftyone
    fiftyone imported
    
    WARNING: using/registering with ALL images.  You should use the --lists option
    From directory  :     /tmp/my_data_set
    Using list file :     None
    Number of images:     ?
    Registering name:     my_dataset
    
    Ignoring unsupported parameter 'label_types' for importer type <class 'fiftyone.utils.data.importers.FiftyOneDatasetImporter'>
    Ignoring unsupported parameter 'load_hierarchy' for importer type <class 'fiftyone.utils.data.importers.FiftyOneDatasetImporter'>
     100% |███████| 200/200 [3.0s elapsed, 0s remaining, 65.3 samples/s]      
    
    ** Let's peek at the first sample - check that it looks ok:**
    
    <Sample: {
        'id': '633d499dad3c137e8ef16292',
        'media_type': 'image',
        'filepath': '/tmp/my_data_set/data/000880.jpg',
        'tags': BaseList(['validation']),
        'metadata': None,
        'ground_truth': <Detections: {
            'detections': BaseList([
                <Detection: {
                    'id': '5f452471ef00e6374aac53c8',
                    'attributes': BaseDict({}),
                    'tags': BaseList([]),
                    'label': 'bird',
                    'bounding_box': BaseList([
                        0.21084375,
                        0.0034375,
                        0.46190625,
                        0.9442083333333334,
                    ]),
                    'mask': None,
                    'confidence': None,
                    'index': None,
                    'area': 73790.37944999996,
                    'iscrowd': 0.0,
                }>,
                <Detection: {
                    'id': '5f452471ef00e6374aac53c9',
                    'attributes': BaseDict({}),
                    'tags': BaseList([]),
                    'label': 'bird',
                    'bounding_box': BaseList([
                        0.74946875,
                        0.489375,
                        0.2164375,
                        0.23183333333333334,
                    ]),
                    'mask': None,
                    'confidence': None,
                    'index': None,
                    'area': 3935.7593000000006,
                    'iscrowd': 0.0,
                }>,
                <Detection: {
                    'id': '5f452471ef00e6374aac53ca',
                    'attributes': BaseDict({}),
                    'tags': BaseList([]),
                    'label': 'bird',
                    'bounding_box': BaseList([
                        0.044234375,
                        0.5282083333333333,
                        0.151390625,
                        0.14145833333333335,
                    ]),
                    'mask': None,
                    'confidence': None,
                    'index': None,
                    'area': 4827.32605,
                    'iscrowd': 0.0,
                }>,
            ]),
        }>,
        'uniqueness': 0.8175834390151201,
        'predictions': <Detections: {
            'detections': BaseList([
                <Detection: {
                    'id': '5f452c60ef00e6374aad9394',
                    'attributes': BaseDict({}),
                    'tags': BaseList([]),
                    'label': 'bird',
                    'bounding_box': BaseList([
                        0.22192673683166503,
                        0.06093006531397502,
                        0.4808845520019531,
                        0.8937615712483724,
                    ]),
                    'mask': None,
                    'confidence': 0.9750854969024658,
                    'index': None,
                }>,
                <Detection: {
                    'id': '5f452c60ef00e6374aad9395',
                    'attributes': BaseDict({}),
                    'tags': BaseList([]),
                    'label': 'bird',
                    'bounding_box': BaseList([
                        0.3962469816207886,
                        0.006943931678930918,
                        0.27418792247772217,
                        0.46793556213378906,
                    ]),
                    'mask': None,
                    'confidence': 0.759726881980896,
                    'index': None,
                }>,
                <Detection: {
                    'id': '5f452c60ef00e6374aad9396',
                    'attributes': BaseDict({}),
                    'tags': BaseList([]),
                    'label': 'bird',
                    'bounding_box': BaseList([
                        0.02506386339664459,
                        0.548487663269043,
                        0.16438478231430054,
                        0.16736234029134114,
                    ]),
                    'mask': None,
                    'confidence': 0.6569182276725769,
                    'index': None,
                }>,
                <Detection: {
                    'id': '5f452c60ef00e6374aad9397',
                    'attributes': BaseDict({}),
                    'tags': BaseList([]),
                    'label': 'bird',
                    'bounding_box': BaseList([
                        0.4889101028442383,
                        0.009576511383056641,
                        0.13802199363708495,
                        0.2093157132466634,
                    ]),
                    'mask': None,
                    'confidence': 0.2359301745891571,
                    'index': None,
                }>,
                <Detection: {
                    'id': '5f452c60ef00e6374aad9398',
                    'attributes': BaseDict({}),
                    'tags': BaseList([]),
                    'label': 'elephant',
                    'bounding_box': BaseList([
                        0.015171945095062256,
                        0.555288823445638,
                        0.1813342332839966,
                        0.15938574473063152,
                    ]),
                    'mask': None,
                    'confidence': 0.221974179148674,
                    'index': None,
                }>,
                <Detection: {
                    'id': '5f452c60ef00e6374aad9399',
                    'attributes': BaseDict({}),
                    'tags': BaseList([]),
                    'label': 'bear',
                    'bounding_box': BaseList([
                        0.017808181047439576,
                        0.5488224665323893,
                        0.17450940608978271,
                        0.16891117095947267,
                    ]),
                    'mask': None,
                    'confidence': 0.1965726613998413,
                    'index': None,
                }>,
                <Detection: {
                    'id': '5f452c60ef00e6374aad939a',
                    'attributes': BaseDict({}),
                    'tags': BaseList([]),
                    'label': 'elephant',
                    'bounding_box': BaseList([
                        0.16558188199996948,
                        0.5723957061767578,
                        0.09993256330490112,
                        0.10098978678385416,
                    ]),
                    'mask': None,
                    'confidence': 0.18904592096805573,
                    'index': None,
                }>,
                <Detection: {
                    'id': '5f452c60ef00e6374aad939b',
                    'attributes': BaseDict({}),
                    'tags': BaseList([]),
                    'label': 'sheep',
                    'bounding_box': BaseList([
                        0.213010573387146,
                        0.05354320605595907,
                        0.5153374671936035,
                        0.8933518091837566,
                    ]),
                    'mask': None,
                    'confidence': 0.11480894684791565,
                    'index': None,
                }>,
                <Detection: {
                    'id': '5f452c60ef00e6374aad939c',
                    'attributes': BaseDict({}),
                    'tags': BaseList([]),
                    'label': 'bird',
                    'bounding_box': BaseList([
                        0.29751906394958494,
                        0.010790024201075237,
                        0.3315577507019043,
                        0.34026527404785156,
                    ]),
                    'mask': None,
                    'confidence': 0.11089690029621124,
                    'index': None,
                }>,
                <Detection: {
                    'id': '5f452c60ef00e6374aad939d',
                    'attributes': BaseDict({}),
                    'tags': BaseList([]),
                    'label': 'elephant',
                    'bounding_box': BaseList([
                        0.08351035118103027,
                        0.5574632008870443,
                        0.18209288120269776,
                        0.1426785151163737,
                    ]),
                    'mask': None,
                    'confidence': 0.0971052274107933,
                    'index': None,
                }>,
                <Detection: {
                    'id': '5f452c60ef00e6374aad939e',
                    'attributes': BaseDict({}),
                    'tags': BaseList([]),
                    'label': 'bird',
                    'bounding_box': BaseList([
                        0.4461814880371094,
                        0.0007838249827424685,
                        0.209574556350708,
                        0.309667714436849,
                    ]),
                    'mask': None,
                    'confidence': 0.08403241634368896,
                    'index': None,
                }>,
                <Detection: {
                    'id': '5f452c60ef00e6374aad939f',
                    'attributes': BaseDict({}),
                    'tags': BaseList([]),
                    'label': 'bird',
                    'bounding_box': BaseList([
                        0.5395165920257569,
                        0.034476550420125325,
                        0.07703280448913574,
                        0.16296254793802897,
                    ]),
                    'mask': None,
                    'confidence': 0.07699568569660187,
                    'index': None,
                }>,
                <Detection: {
                    'id': '5f452c60ef00e6374aad93a0',
                    'attributes': BaseDict({}),
                    'tags': BaseList([]),
                    'label': 'bear',
                    'bounding_box': BaseList([
                        0.217216157913208,
                        0.05954849322636922,
                        0.49451656341552735,
                        0.8721434275309244,
                    ]),
                    'mask': None,
                    'confidence': 0.058097004890441895,
                    'index': None,
                }>,
                <Detection: {
                    'id': '5f452c60ef00e6374aad93a1',
                    'attributes': BaseDict({}),
                    'tags': BaseList([]),
                    'label': 'sheep',
                    'bounding_box': BaseList([
                        0.018094074726104737,
                        0.5562847137451172,
                        0.17362892627716064,
                        0.15742950439453124,
                    ]),
                    'mask': None,
                    'confidence': 0.0519101656973362,
                    'index': None,
                }>,
            ]),
        }>,
    }>
    


here ``--type`` depends on the directory/file structure your data
directory has. Typical values are
``FiftyOneDataset, OpenImagesV6Dataset, ImageDirectory``. Please take a
look in
`here <https://voxel51.com/docs/fiftyone/api/fiftyone.types.dataset_types.html>`__
for more information.

Let’s check that the dataset got registered correctly:

.. code:: bash

    compressai-vision list


.. code-block:: text

    importing fiftyone
    fiftyone imported
    
    datasets currently registered into fiftyone
    name, length, first sample path
    mpeg-vcm-detection, 5000, /home/sampsa/fiftyone/mpeg-vcm-detection/data
    mpeg-vcm-detection-dummy, 1, /home/sampsa/fiftyone/mpeg-vcm-detection/data
    mpeg-vcm-segmentation, 5000, /home/sampsa/fiftyone/mpeg-vcm-segmentation/data
    my_dataset, 200, /tmp/my_data_set/data
    open-images-v6-validation, 8189, /home/sampsa/fiftyone/open-images-v6/validation/data
    quickstart, 200, /home/sampsa/fiftyone/quickstart/data


A more detailed look into the dataset:

.. code:: bash

    compressai-vision show --dataset-name=my_dataset


.. code-block:: text

    importing fiftyone
    fiftyone imported
    
    dataset info:
    Name:        my_dataset
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
    
    test-loading first image from /tmp/my_data_set/data/000880.jpg
    loaded image with dimensions (480, 640, 3) ok


Let’s deregister the dataset:

.. code:: bash

    compressai-vision deregister --y --dataset-name=my_dataset


.. code-block:: text

    importing fiftyone
    fiftyone imported
    removing dataset my_dataset from fiftyone


Check it got removed:

.. code:: bash

    compressai-vision list


.. code-block:: text

    importing fiftyone
    fiftyone imported
    
    datasets currently registered into fiftyone
    name, length, first sample path
    mpeg-vcm-detection, 5000, /home/sampsa/fiftyone/mpeg-vcm-detection/data
    mpeg-vcm-detection-dummy, 1, /home/sampsa/fiftyone/mpeg-vcm-detection/data
    mpeg-vcm-segmentation, 5000, /home/sampsa/fiftyone/mpeg-vcm-segmentation/data
    open-images-v6-validation, 8189, /home/sampsa/fiftyone/open-images-v6/validation/data
    quickstart, 200, /home/sampsa/fiftyone/quickstart/data


Let’s remove the image data as well:

.. code:: ipython3

    rm -rf /tmp/my_data_set

