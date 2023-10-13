In this chapter you will learn:

-  to evaluate a custom model you’re currently developing agains the
   mpeg-vcm tests

As in the previous chapters, let’s first check we have the dataset
``oiv6-mpeg-detection-v1`` available:

.. code:: bash

    compressai-vision list


.. code-block:: text

    importing fiftyone
    fiftyone imported
    
    datasets currently registered into fiftyone
    name, length, first sample path
    detectron-run-sampsa-oiv6-mpeg-detection-v1-2022-11-16-17-22-40-319395, 2, /home/sampsa/fiftyone/oiv6-mpeg-detection-v1/data
    detectron-run-sampsa-oiv6-mpeg-detection-v1-2022-11-16-17-24-14-478278, 2, /home/sampsa/fiftyone/oiv6-mpeg-detection-v1/data
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


In order for your custom model to work with compressai-vision, it needs
to be in a separate folder. The entry-point must be called ``model.py``.
We provide an example for this: please take a look at the
`examples/models/bmshj2018-factorized/ <https://github.com/InterDigitalInc/CompressAI-Vision/tree/main/examples/models/bmshj2018-factorized>`__
folder, where you have the following files:

::

   ├── bmshj2018-factorized-prior-1-446d5c7f.pth.tar
   ├── bmshj2018-factorized-prior-2-87279a02.pth.tar
   └── model.py

The ``.pth.tar`` files are the checkpoints of your model, while
``model.py`` contains the pytorch/compressai custom code of your model.

The requirement for ``model.py`` is simple. You need to define this
function:

::

   getEncoderDecoder(quality=None, **kwargs)

which returns a subclass of an ``EncoderDecoder`` instance.
``EncoderDecoder`` objects know how to do just that: encode and decode
an image, and returning the final decoded image with bits-per-pixel
value.

``quality`` should be an integer parameter (it will be used by the
``--qpars`` command-line flag). The quality parameter is mapped to a
certain model checkpoint file in ``model.py``:

::

   qpoint_per_file = {
       1 : "bmshj2018-factorized-prior-1-446d5c7f.pth.tar",
       2 : "bmshj2018-factorized-prior-2-87279a02.pth.tar"
   }

i.e. if you define ``--qpars=1``, the model will use
``bmshj2018-factorized-prior-1-446d5c7f.pth.tar`` from the directory.

As you can learn from the code, ``EncoderDecoder`` object uses an
underlying (compressai-based) model to perform the actual
encoding/decoding.

The requirement for the model class
(``class FactorizedPrior(CompressionModel)`` in the example model.py)
are minimal: your model class should have two methods, called
``compress`` and ``decompress``. Method ``compress`` takes in an RGB
image tensor and returns bitstream, while ``decompress`` takes in
bitstream and returns a recovered image.

The exact signatures are:

::

   def compress(self, x): -> dict
       # where x is a torch RGB image tensor (batch, 3, H, W) 
       ...
       return {"strings": STRINGS, "shape": SHAPE}
       # STRINGS: a list where STRINGS[0][0] is a bytes object (the encoded bitstream) and SHAPE is some shape information used by your model
       
       
   def decompress(self, STRINGS, SHAPE): -> dict
       # where STRINGS and SHAPE are the objects returned by compress
       ...
       return {"x_hat": x_hat}
       # where x_hat is a torch RGB image tensor (batch, 3, H, W)

This signature/interface is used by the compressai library models. When
you have these same methods in *your* custom model, you can use the
``CompressAIEncoderDecoder`` straight out of the box with it.

You can also implement your own ``EncoderDecoder`` class (say, for
comparing results to classical codecs like jpeg, etc.). For this, please
refer to the example jpeg ``EncoderDecoder`` class in the library
tutorial.

So, take a copy of the ``examples/models/bmshj2018-factorized/`` folder
into your disk and run:

::

   compressai-vision detectron2-eval --y \
   --dataset-name oiv6-mpeg-detection-v1 \
   --slice=0:2 \
   --scale=100 \
   --gt-field=detections \
   --eval-method=open-images \
   --progressbar \
   --compression-model-path /path/to/examples/models/bmshj2018-factorized/ \
   --qpars=1,2 \
   --output=detectron2_bmshj2018-factorized.json \
   --model=COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x.yaml

This will evaluate your custom model with Detectron2.

Again, for an actual production run, you would remove the ``--slice``
argument.

