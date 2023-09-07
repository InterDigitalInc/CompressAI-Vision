<p align="center">
  <img src="docs/source/_static/logo.svg" alt="CompressAI-Vision-logo">
</p>

# CompressAI-FCVCM

CompressAI-FCVCM is an evaluation framework for compressing intermediate features produced in the context of split models.
It helps you to develop, test and evaluate compression models with standardized tests in the context of MPEG "Feature Compression for Video Coding for Machines" (FCVCM).

The figure below shows the main framework
<p align="center">
  <img src="docs/source/media/images/fcvcm-scope.png" alt="split model evaluation pipeline">
</p>


## Documentation

A complete documentation is provided [here](https://interdigitalinc.github.io/CompressAI-Vision/index.html)

## General Prerequisites
pyhton3, cmake, patch, gcc/g++, ffmpeg

## installation

To get started locally and install the development version of CompressAI-FCVCM, first create and activate a [virtual environment](https://docs.python.org/3.8/library/venv.html) with python>=3.8:

Bash scripts are provided to get proper installation of dependencies. First, if you want to manually export CUDA related paths, please source (e.g. for CUDA 11.8):
```
bash scripts/env_cuda.sh 11.8
```

Then, install the different models and related dependencies (with default versions) using
```
bash scripts/install.sh
```


For more otions, check:
```
bash scripts/install.sh --help
```

To install CompressAI-Vision:
```
pip install -e .
```