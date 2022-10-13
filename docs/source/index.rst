CompressAIVision
================

CompressAI-Vision helps you to develop, test and evaluate compression models with standardized tests in the context of "Video Coding for Machines" (VCM), i.e. compression methods optimized for machine tasks algorithms such as Neural-Network (NN)-based detectors.


.. mermaid::

   graph LR
      A[input<br>image/video]:::other -->B
      A --> B1
      B[Traditional<br>Encoder]:::other --> C
      B1[NN<br>Encoder]:::cai--> C
      C[Bitstream]:::other --> D
      C --> D1
      D[Traditional<br>Decoder]:::other--> E
      D1[NN<br>Decoder]:::cai-->E
      D1 -.-> H
      E[Reconstructed<br>media]:::other--> G
      E --> F1
      G[Visual<br>Quality<br>Metrics]:::cai
      F1[Detector]:::cav --> G1[Task<br>Metrics]:::cav
      H[Feature<br>Maps]:::future -.-> F1
      classDef cai stroke:#63C5DA,stroke-width:4px
      classDef cav stroke:#FFC0CB,stroke-width:4px
      classDef other stroke:#008000,stroke-width:4px
      classDef future stroke:#FFBF00,stroke-width:4px

A typical metric for evaluating the encoder's efficiency for serving a detection/segmentation task,
is the mean average precision (mAP) as a function of encoding/quality parameters:


For testing the VCM pipeline, various mAP measures (as in COCO or OpenImageV6 evaluation protocols) and datasets can be used,
while the deep-learning CompressAI-based models are typically compared againts a well-known "anchor" pipeline, featuring a
classical image/video codec like H266/VVC.


You can use a single command-line interface (CLI) tool to evaluate your custom deep-learning model according to the
standards set up by the MPEG-VCM working group.

Some features/details of CompressAI-Vision:

- It uses `fiftyone <https://voxel51.com/docs/fiftyone/>`_ for dataset downloading, handling, visualization and evaluation protocols.  Several evaluation protocols are supported via fiftyone

- Supports `CompressAI <https://interdigitalinc.github.io/CompressAI>`_,  `VTM <https://vcgit.hhi.fraunhofer.de/jvet/VVCSoftware_VTM>`_ and custom modules for the encoding/decoding part

- Uses `Detectron2 <https://detectron2.readthedocs.io/en/latest/index.html>`_ for detector and image segmentation models

- Docker images, including all required components (CompressAI, VTM, Detectron2, CUDA support and whatnot) are provided

- CompressAI-Vision supports parts of the Common Test Conditions defined by the ISO/MPEG VCM Ad-hoc Group, including standardized datasets (typically OpenImageV6 subsets), evaluation protocols (OpenImageV6) and anchor pipelines based on the compression using the state-of-the-art H.266/VCC codec.

To get started, please go to through the installation steps and then do the CLI tutorial.

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
  :caption: faq

  faq

.. toctree::
   :caption: Development

   Github repository <https://github.com/InterDigitalInc/CompressAI-Vision>
