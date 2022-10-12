CompressAIVision
================

CompressAI-Vision helps you to develop, test and evaluate
`CompressAI <https://interdigitalinc.github.io/CompressAI>`_
models with standardized tests.

Image and video coding models can be tested using various metrics, say SSIM and PNSR, but also against image detection 
and segmentation tasks.

Developing optimized encoders for pipelines including deep-learning-based detectors is called 
"video coding for machines" (**VCM**) and its goal is to create
efficient encoders for machine learning tasks. Following is a pipeline of video/image compression for machine vision task.
End-to-end compression model for human consumption (components in blue box) 
are implemented with Interdigital CompressAI library. 
Our library is a companion of CompressAI and implements 
the computer vision task and corresponding evaluations (pink boxes). 
Furthermore, we also provide support for traditional codec (for example, VTM codec) 
so the user can benchmark their model with the performance of traditional codec. 
In the future, we are also going to include support of using feature map as input of computer vision tasks.


.. mermaid::

   graph LR
      A[input video/image]:::other -->B
      A --> B1
      B[Traditional Encoder]:::other --> C
      B1[E2E Encoder]:::cai--> C
      C[Compressed bitstream]:::other --> D
      C --> D1
      D[Traditional Decoder]:::other--> E
      D1[E2E Decoder]:::cai-->E
      D1 -.-> H
      E[Reconstructed video/image]:::other--> F
      E --> F1
      F[Human consumption]:::cai --> G[Visual Quality Metrics]:::cai
      F1[Computer vision task]:::cav --> G1[Task Metrics]:::cav
      H[Feature map]:::future -.-> F1
      classDef cai stroke:#63C5DA,stroke-width:4px
      classDef cav stroke:#FFC0CB,stroke-width:4px
      classDef other stroke:#008000,stroke-width:4px
      classDef future stroke:#FFBF00,stroke-width:4px


A typical metric for evaluating the encoder's efficiency for serving a detection/segmentation task, 
is the mean average precision (mAP) as a function of encoding/quality parameters:

TODO: add a figure here

For testing the VCM pipeline, various mAP measures (as in COCO or OpenImageV6 evaluation protocols) and datasets can be used, 
while the deep-learning CompressAI-based models are typically compared againts a well-known "anchor" pipeline, featuring a 
classical image/video codec, say, H266.

The **MPEG working group for VCM** has created a set of standards for testing VCM, including standardized data/image sets 
(typically OpenImageV6 subsets), evaluation protocols (OpenImageV6) and anchor pipelines (H266 based)

CompressAI-Vision makes the handling and evaluation of the mentioned datasets with any encoding/decoding 
and detection pipeline a breeze.

You can use a single command-line interface (CLI) tool to evaluate your custom deep-learning model according to the
standards set up by the MPEG-VCM working group.

Some features/details of CompressAI-Vision:

- It uses `fiftyone <https://voxel51.com/docs/fiftyone/>`_ for dataset downloading, handling, visualization and evaluation protocols.  Several evaluation protocols are supported via fiftyone

- Supports `CompressAI <https://interdigitalinc.github.io/CompressAI>`_,  `VTM <https://vcgit.hhi.fraunhofer.de/jvet/VVCSoftware_VTM>`_ and custom modules for the encoding/decoding part

- Uses `Detectron2 <https://detectron2.readthedocs.io/en/latest/index.html>`_ for detector and image segmentation models

- Supports official MPEG committee input files (TODO: link to MPEG/VCM document?)

- Docker images, including all required components (CompressAI, VTM, Detectron2, CUDA support and whatnot) are provided to get you started asap

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
  :caption: Utils

  cli_usage

.. toctree::
  :maxdepth: 2
  :caption: faq

  faq

.. toctree::
   :caption: Development

   Github repository <https://github.com/InterDigitalInc/CompressAI-Vision>
