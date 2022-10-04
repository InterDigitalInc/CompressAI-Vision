CLI Tutorial
============

.. toctree::
   :maxdepth: 2
   :caption: CLI HOWTO

   cli_tutorial_1


Library Tutorial
================

In order to test your encoder's performance against the official MPEG/VCM working group input files,
you need first to convert the input files into a standard/recognized format.  After this, you can use them with any standard tool.

We'll be using fiftyone to store, manage, visualize and evaluate datasets.  So the next thing is to import the data into fiftyone.
Fiftyone is then used to evaluate your results which have been generated using Detectron2 and your encoder/decoder.

So, first of all, you should take a look at `fiftyone <https://voxel51.com/docs/fiftyone/user_guide>`_ and therein, at least
how convenient `fiftyone datasets <https://voxel51.com/docs/fiftyone/user_guide/using_datasets.html#>`_ are.

Fiftyone datasets are particularly convenient: fiftyone starts silently a mongodb server on the background, providing a single
source of ground truths and a place to save your detection results, visible to all python processes running on the same host.

Please follow rigorously the steps (1)-(4) in the tutorial list below.

.. toctree::
   :maxdepth: 2
   :caption: LIB HOWTO

   download
   convert
   detectron2
   evaluate
   encdec

There are also convenient cli utilities to achieve steps (1)-(4) in single-shot commands.  Please see :ref:`here <cli>` for more info.
