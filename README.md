# CompressAI-Vision

*for complete documentation, please go directly to (TODO: link)*

## Synopsis

CompressAI-Vision helps you to develop, test and evaluate [CompressAI](https://interdigitalinc.github.io/CompressAI) models with standardized tests.

Image and video coding models can be tested using various metrics, say SSIM and PNSR, but also againts image detection and segmentation tasks.  

Developing optimized encoders for pipelines including deep-learning-based detectors is called "video coding for machines" (**VCM**) and its goal is to create efficient encoders for machine learning tasks:
```
Video stream --> Encoding --> bitstream over internet --> Decoding --> Detector
```

TODO: Jacky had some nice diagrams for this..?

A typical metric for evaluating the encoder's efficiency for serving a detection/segmentation task, is the mean average precision (mAP) as a function of encoding/quality parameters:

TODO: add a figure here

For testing the VCM pipeline, various mAP measures (as in COCO or OpenImageV6 evaluation protocols) and datasets can be used, while the deep-learning CompressAI-based models are typically compared againts a well-known "anchor" pipeline, featuring a classical image/video codec, say, H266.

The **MPEG working group for VCM** has created a set of standards for testing VCM, including standardized data/image sets (typically OpenImageV6 subsets), evaluation protocols (OpenImageV6) and anchor pipelines (H266 based)

## Features

CompressAI-Vision makes the handling and evaluation VCM with any encoding/decoding pipeline a breeze:

- It uses [fiftyone](https://voxel51.com/docs/fiftyone/) for dataset downloading, handling, visualization and evaluation protocols (fiftyone supports several evaluation protocols)

- Supports [CompressAI](https://interdigitalinc.github.io/CompressAI), [VTM](https://vcgit.hhi.fraunhofer.de/jvet/VVCSoftware_VTM) and custom modules for the encoding/decoding part

- Uses [Detectron2](https://detectron2.readthedocs.io/en/latest/index.html) for detector and image segmentation models

- Supports official MPEG committee input files (TODO: link to mpeg vcm document?)

- Single-shot CLI commands for fast input file import, image download and evaluation

- Docker images, including all required components (CompressAI, VTM, Detectron2, CUDA support and whatnot) are provided to get you started asap

Tutorials, API documentation and notebook examples are provided so please, go to the documentation to get started (TODO: link here)

## Installation

The software stack looks like this (all with CUDA support):

- [PyTorch](https://pytorch.org/)
- [CompressAI](https://interdigitalinc.github.io/CompressAI)
- [Detectron2](https://detectron2.readthedocs.io/en/latest/index.html)
- [fiftyone](https://voxel51.com/docs/fiftyone/)
- [VTM](https://vcgit.hhi.fraunhofer.de/jvet/VVCSoftware_VTM)
- _This_ library (CompressAI-Vision)

Instructions for creating a correct ``virtualenv`` are provided in the documentation (TODO: link).

Docker images including the software stack are also provided (TODO: link).

## For developers

### Testing

Until a proper test pipeline is established, for the absolute minimal testing, you can use this command to see that's nothing accutely broken:
```
compressai-vision-info
```

### Code formatting

Code is formatted using black (install with ``pip3 install --user black``).

The CI pipeline checks for your code formatting, so be sure that it conforms to black before committing.  To do that, run (in this directory):
```
black --check --diff compressai_vision
```
To apply the formatting, run
```
black compressai_vision
```
You might want to install the "black formatter" extension if you're into VSCode.

### Compiling documentation

You need to install the [furo theme](https://github.com/pradyunsg/furo).  A good idea is to install it into the same virtualenv as all the other stuff.

You need also this:
```
sudo apt-get install pandoc
```

Tutorials are produced from notebooks that are in [docs/source/tutorials](docs/source/tutorials).  If you update the notebooks, first you need to run ``compile.bash`` therein.  

To produce the html documentation, run in [docs/](docs/):
```
make html
```
The go with your browser to [docs/index.html](docs/index.html)

## Copyright

InterDigital 2022

## License

TODO

## Authors

Sampsa Riikonen

Jacky Yat-Hong Lam

Fabien Racapé
