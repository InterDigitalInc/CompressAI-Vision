In this chapter you will learn:

-  to evaluate a custom model you’re currently developing agains the
   mpeg-vcm tests

As in the previous chapters, let’s first check we have the dataset
``mpeg-vcm-detection`` available:

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


In order for your custom model to work with compressai-vision, it needs
to be in a separate folder. The entry-point must be called ``model.py``.
We provide an example for this: please take a look at the
`examples/models/bmshj2018-factorized/ <https://github.com/InterDigitalInc/CompressAI-Vision-Internal/tree/main/examples/models/bmshj2018-factorized>`__
folder, where you have the following files:

::

   ├── bmshj2018-factorized-prior-1-446d5c7f.pth.tar
   ├── bmshj2018-factorized-prior-2-87279a02.pth.tar
   ├── __init__.py
   ├── model.py
   └── utils.py

The ``.pth.tar`` files are the checkpoints of your model, while
``model.py`` contains the pytorch/compressai custom code for your model.
Other python files are whatever python code your custom model might
require.

The requirement for ``model.py`` is simple. You simply define this
function:

::

   getModel(quality=None, pretrained=False, **kwargs)

which returns an instance of your model class. ``quality`` should be an
integer parameter (it will be used by the ``--qpars`` command-line
flag).

The requirement for the model class
(``class FactorizedPrior(CompressionModel)`` in the example model.py)
are also minimal. Your model class should have two methods, called
``compress`` and ``decompress``: ``compress`` takes in an RGB image
tensor and returns bitstream, while ``decompress`` takes in bitstream
and returns a recovered image.

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

These signatures/interface is used by the compressai library models and
they are used by the ``CompressAIEncoderDecoder`` class (see the
tutorial on creating a EncoderDecoder class).

If you want to use another kind of API, you need to define your own
``EncoderDecoder`` class. Please refer to the example jpeg
``EncoderDecoder`` class in the tutorial.

TODO: in model.py user can define their own EncoderDecoder class if needed?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

So, take a copy of the ``examples/models/bmshj2018-factorized/`` folder
into your disk and then run:

::

   compressai-vision detectron2-eval --y \
   --dataset-name mpeg-vcm-detection \
   --slice=0:2 \
   --scale=100 \
   --gt-field=detections \
   --eval-method=open-images \
   --progressbar \
   --compression-model-path /path/to/examples/models/bmshj2018-factorized/ \
   --compression-model-checkpoint /path/to/examples/models/bmshj2018-factorized/bmshj2018-factorized-prior-1-446d5c7f.pth.tar \
   --output=detectron2_bmshj2018-factorized.json \
   --model=COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x.yaml

This will evaluate your custom model with Detectron2. Note that here the
``--qpars`` quality parameters have been substituted with your custom
model’s checkpoints.

TODO: we want to have the possibility to use integer qpars also for custom models, right? The model.py allows for this (see the definition of getModel() up there)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Again, for an actual production run, you would remove the ``--slice``
argument.