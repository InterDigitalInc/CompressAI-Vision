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


