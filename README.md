# CompressAI-Vision

*for complete documentation, please go directly to (TODO: link)*

## Synopsis

CompressAI-Vision helps you to develop, test and evaluate [CompressAI](https://interdigitalinc.github.io/CompressAI) models with standardized tests.

Image and video coding models can be tested using various metrics, say SSIM and PNSR, but also againts image detection and segmentation tasks.  

Developing optimized encoders for pipelines including deep-learning-based detectors is called "video coding for machines" (VCM) and its goal is to create efficient encoders for machine learning tasks:
```
Video stream --> Encoding + Decoding --> Detector
```

A typical metric for evaluating the encoder's efficiency for serving a detection/segmentation task, is the mean average precision (mAP) as function of a set of encoding/quality parameters:

TODO: add a figure here

For testing the VCM pipeline, various mAP measures (as in COCO or OpenImageV6 evaluation protocols) and datasets can be used, while the deep-learning CompressAI-based models are typically compared againts a well-known "anchor" pipeline, featuring a classical/standard image/video codec, say, H266.

The **MPEG working group for VCM** has created a set of standards for testing VCM, including standardized data/image sets (typically OpenImageV6 subsets), evaluation protocols (OpenImageV6) and anchor pipelines (VTM/H266 based)

## Features

CompressAI-Vision makes the handling and evaluation of the mentioned datasets with any encoding/decoding and detection pipeline a breeze:

- It uses [fiftyone](https://voxel51.com/docs/fiftyone/) for dataset downloading, handling, visualization and evaluation protocols.  Several evaluation protocols are supported via fiftyone

- Supports CompressAI, VTM/H266 (TODO: link) and custom modules for the encoding/decoding part

- Uses [Detectron2](https://detectron2.readthedocs.io/en/latest/index.html) for detector and image segmentation models

- Supports official MPEG committee input files (TODO: link to mpeg vcm document?)

- Single-shot CLI commands for fast input file import, image download and evaluation

- Docker images, including all required components (CompressAI, VTM, Detectron2, CUDA support and whatnot) are provided to get you started asap

Tutorials, API documentation and notebook examples are provided, so please, go to the documentation to get started (TODO: link here)

## Installation

Your software stack will look like this (all with CUDA support):

- [PyTorch](https://pytorch.org/)
- [CompressAI](https://interdigitalinc.github.io/CompressAI)
- [Detectron2](https://detectron2.readthedocs.io/en/latest/index.html)
- [fiftyone](https://voxel51.com/docs/fiftyone/)
- VTM (TODO: link)
- _This_ library (CompressAI-Vision)
- Jupyter notebooks

Instructions for creating a correct ``virtualenv`` are provided in the documentation (TODO: link).

Docker images with the correct software stack are also provided (TODO: link).

## Copyright

InterDigital 2022

## License

TODO

## Authors

Sampsa Riikonen

Jacky Lam

Fabien
