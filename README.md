<p align="center">
  <img src="docs/source/_static/logo.svg" alt="CompressAI-Vision-logo">
</p>

CompressAI-Vision helps you to develop, test and evaluate compression models with standardized tests in the context of compression methods optimized for machine tasks algorithms such as Neural-Network (NN)-based detectors.

It currently focuses on two types of pipeline:

- Video compression for remote inference (`compressai-remote-inference`), which corresponds to the MPEG "Video Coding for Machines" (VCM) activity.

- Split inference (`compressai-split-inference`), which  includes an evaluation framework for compressing intermediate features produced in the context of split models. The software supports all thepipelines considered in the related MPEG  activity: "Feature Compression for Machines" (FCM).

<p align="center">
  <img src="docs/source/media/images/compressai-vision-pipelines.png" alt="CompressAI-Vision supported pipelines">
</p>

## Features

- [Detectron2](https://detectron2.readthedocs.io/en/latest/index.html) for Object Detection (Faster-RCNN) and Instance Segmentation (Mask-RCNN)

- [JDE](https://github.com/Zhongdao/Towards-Realtime-MOT) for Object Tracking

- [YOLOX-Darknet53](https://github.com/Megvii-BaseDetection/YOLOX) for Object Detection

- [MMPOSE RTMO](https://github.com/open-mmlab/mmpose/tree/main/projects/rtmo) for Pose Estimation (Bottom Up)

- [Segment Anything](https://github.com/facebookresearch/segment-anything/tree/main)

## Documentation

A complete documentation is provided [here](https://interdigitalinc.github.io/CompressAI-Vision/index.html), including [installation](https://interdigitalinc.github.io/CompressAI-Vision/installation), [CLI usage](https://interdigitalinc.github.io/CompressAI-Vision/cli_usage.html), as well as [tutorials](https://interdigitalinc.github.io/CompressAI-Vision/tutorials).

## installation

The CompressAI library providing learned compresion modules is available as a submodule. It can be initilized by running:
```
git submodule update --init --recursive
```
Note: the installation scripts documented below installs compressai from source expects the submodule to be populated. 

CompressAI-Vision can be installed using a virtual environment and pip or using uv. 

### 1. Using a virtual environment:

#### Initialization of the environment
To get started locally and install the development version of CompressAI-Vision, first create a [virtual environment](https://docs.python.org/3.8/library/venv.html) with python==3.8:

```
python3.8 -m venv venv
source ./venv/bin/activate
pip install -U pip
```

#### Installation of compressai-vision and supported vision models:

First, if you want to manually export CUDA related paths, please source (e.g. for CUDA 11.8):
```
bash scripts/env_cuda.sh 11.8
```
Then, please run:
```
bash scripts/install.sh 
```

To install the dependencies in conformance with MPEG FCM Test Conditions, run:
```
bash scripts/install.sh --fcm-cttc (--cpu)
```


For more otions, check:
```
bash scripts/install.sh --help
```

NOTE 1: install.sh gives you the possibility to install vision models' source and weights at specified locations so that mutliple versions of compressai-vision can point to the same installed vision models

NOTE 2: the downlading of JDE pretrained weights might fail. Check that the size of following file is ~558MB.
path/to/weights/jde/jde.1088x608.uncertainty.pt
The file can be downloaded at the following link (in place of the above file path):
"https://docs.google.com/uc?export=download&id=1nlnuYfGNuHWZztQHXwVZSL_FvfE551pA"

### 2. Using uv:
Within the root folder of compressai-vision:
```
bash scripts/install_uv.sh
```

Note: Make sure you pin the desired installed python version before, e.g., 
```
uv python pin 3.8
```

## Usage

### Split inference pipelines

To run split-inference pipelines, please use the following command:
```
compressai-split-inference --help
```

Note that the following entry point is kept for backward compability. It runs split inference as well. 
```
compressai-vision-eval --help
```


For example for testing a full split inference pipelines without any compression, run

```
compressai-vision-eval --config-name=eval_split_inference_example
```

### Remote inference pipelines

For remote inference (MPEG VCM-like) pipelines, please run:
```
compressai-remote-inference --help
```

### Configurations

Please check other configuration examples provided in ./cfgs as well as examplary scripts in ./scripts

Test data related to the MPEG FCM activity can be found in ./data/mpeg-fcm/

## For developers

After your dev, you can run (and adapt) test scripts from the scripts/tests directory. Please check [scripts/tests/README.md] for more details

### Contributing

Code is formatted using black and isort. To format code, type:
```
make code-format
```
Static checks with those same code formatters can be run manually with:
```
make static-analysis
```

### Compiling documentation

To produce the html documentation, from [docs/](docs/), run:
```
make html
```
To check the pages locally, open [docs/_build/html/index.html](docs/index.html)

## License

CompressAI-Vision is licensed under the BSD 3-Clause Clear License

## Authors

Fabien Racapé, Hyomin Choi, Eimran Eimon, Sampsa Riikonen, Jacky Yat-Hong Lam

## Citation

If you use this project, please cite:

@article{compressai_vision,
  title={CompressAI-Vision: Open-source software to evaluate compression methods for computer vision tasks},
  author={Choi, Hyomin and Han, Heeji and Rosewarne, Chris and Racap{\'e}, Fabien},
  journal={arXiv preprint arXiv:2509.20777},
  year={2025}
}

## Related links
 * [HEVC HM reference software](https://hevc.hhi.fraunhofer.de)
 * [VVC VTM reference software](https://vcgit.hhi.fraunhofer.de/jvet/VVCSoftware_VTM)
 * [Detectron2](https://detectron2.readthedocs.io/en/latest/index.html)
 * [JDE](https://github.com/Zhongdao/Towards-Realtime-MOT.git)
 * [YOLOX](https://github.com/Megvii-BaseDetection/YOLOX)
 * [MMPOSE RTMO](https://github.com/open-mmlab/mmpose/tree/main/projects/rtmo)
 * [Segment Anything](https://github.com/facebookresearch/segment-anything/tree/main)