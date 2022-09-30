CompressAIVision
================

CompressAI-Vision helps you to develop, test and evaluate
`CompressAI <https://interdigitalinc.github.io/CompressAI>`_
models with standardized tests.

Image and video coding models can be tested using various metrics, say SSIM and PNSR, but also againts image detection and segmentation tasks.

Developing optimized encoders for pipelines including deep-learning-based detectors is called "video coding for machines" (**VCM**) and its goal is to create
efficient encoders for machine learning tasks:

::

    Video stream --> Encoding --> bitstream over internet --> Decoding --> Detector

TODO: Jacky had nice diagrams for this..?

A typical metric for evaluating the encoder's efficiency for serving a detection/segmentation task, is the mean average precision (mAP)
as a function of encoding/quality parameters:

TODO: add a figure here

For testing the VCM pipeline, various mAP measures (as in COCO or OpenImageV6 evaluation protocols) and datasets can be used, while the deep-learning
CompressAI-based models are typically compared againts a well-known "anchor" pipeline, featuring a classical image/video codec, say, H266.

The **MPEG working group for VCM** has created a set of standards for testing VCM, including standardized data/image sets (typically OpenImageV6 subsets),
evaluation protocols (OpenImageV6) and anchor pipelines (H266 based)

CompressAI-Vision makes the handling and evaluation of the mentioned datasets with any encoding/decoding and detection pipeline a breeze:

- It uses `fiftyone <https://voxel51.com/docs/fiftyone/>`_ for dataset downloading, handling, visualization and evaluation protocols.  Several evaluation protocols are supported via fiftyone

- Supports `CompressAI <https://interdigitalinc.github.io/CompressAI>`_,  `VTM <https://vcgit.hhi.fraunhofer.de/jvet/VVCSoftware_VTM>`_ and custom modules for the encoding/decoding part

- Uses `Detectron2 <https://detectron2.readthedocs.io/en/latest/index.html>`_ for detector and image segmentation models

- Supports official MPEG committee input files (TODO: link to MPEG/VCM document?)

- Single-shot CLI commands for fast input file import, image download and evaluation

- Docker images, including all required components (CompressAI, VTM, Detectron2, CUDA support and whatnot) are provided to get you started asap

.. toctree::
   :maxdepth: 2
   :caption: Setup

   installation
   docker

.. toctree::
   :maxdepth: 2
   :caption: Tutorials

   tutorials/index.rst

.. toctree::
   :maxdepth: 2
   :caption: Library API

   conversion/index.rst
   evaluation/index.rst

.. toctree::
  :maxdepth: 2
  :caption: Utils

  cli_usage

.. toctree::
  :maxdepth: 2
  :caption: faq

  faq

.. toctree::
   :caption: Development

   Github repository <https://github.com/InterDigitalInc/CompressAI-Vision>
