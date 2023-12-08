Tutorials VCM
=============


Fiftyone
~~~~~~~~

CompressAI-Vision uses fiftyone/mongodb in managing dataset.  Here we take a quick look at fiftyone.

.. toctree::
   :maxdepth: 2
   :caption: FIFTYONE HOWTO

   fiftyone

CLI Tutorial
~~~~~~~~~~~~

CompressAI-Vision has a rich command-line interface for performing all necessary operations for evaluating your deep-learning
encoder/decoder according to the standards set by the MPEG-VCM working group.

.. toctree::
   :maxdepth: 2
   :caption: CLI HOWTO

   cli_tutorial_1
   cli_tutorial_2
   cli_tutorial_3
   cli_tutorial_4
   cli_tutorial_5
   cli_tutorial_6
   cli_tutorial_7

Library Tutorial
~~~~~~~~~~~~~~~~

Here we take a look how to use CompressAI-Vision as a python library.  This is advanced topic.  Normally, the CLI is all you need.

.. toctree::
   :maxdepth: 2
   :caption: LIB HOWTO

   download
   detectron2
   evaluate
   encdec


Input file conversion
~~~~~~~~~~~~~~~~~~~~~

:download:`[download tutorial as notebook]<convert_nb.ipynb>`

The MPEG/VCM working group provides annotations and segmentation data in non-standard format.

We convert this data into OpenImageV6 format and also register it into fiftyone.

.. include:: convert_nb.rst
