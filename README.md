<p align="center">
  <img src="docs/source/_static/logo.svg" alt="CompressAI-Vision-logo">
</p>

CompressAI-Vision helps you to develop, test and evaluate compression models with standardized tests in the context of compression methods optimized for machine tasks algorithms such as Neural-Network (NN)-based detectors.

It currently focuses on two types of pipeline:

- Video compression for remote analysis, which corresponds to the MPEG "Video Coding for Machines" (VCM) activity.

- Split inference, which  includes an evaluation framework for compressing intermediate features produced in the context of split models. The software supports all thepipelines considered in the related MPEG  activity: "Feature Compression for Video Coding for Machines" (FCVCM).

The figure below shows the split inference pipeline:

<p align="center">
  <img src="docs/source/media/images/fcvcm-scope.png" alt="split model evaluation pipeline">
</p>

## Features

- [Detectron2](https://detectron2.readthedocs.io/en/latest/index.html) is used for object detection (Faster-RCNN) and instance segmentation (Mask-RCNN)

- JDE is used for Object Tracking

## Documentation

A complete documentation is provided [here](https://interdigitalinc.github.io/CompressAI-Vision/index.html), including [installation](https://interdigitalinc.github.io/CompressAI-Vision/installation), [CLI usage](https://interdigitalinc.github.io/CompressAI-Vision/cli_usage.html), as well as [tutorials](https://interdigitalinc.github.io/CompressAI-Vision/tutorials).

## installation

To get started locally and install the development version of CompressAI-Vision, first create a [virtual environment](https://docs.python.org/3.8/library/venv.html) with python==3.8:

To install the models relevant for the FCVCM (feature compression):
- First, if you want to manually export CUDA related paths, please source (e.g. for CUDA 11.8):
```
bash scripts/env_cuda.sh 11.8
```
Then, run:, please run:
```
bash scripts/install.sh
```

For more otions, check:
```
bash scritps/install.sh --help
```

The software stack includes (all with CUDA support):

- [PyTorch](https://pytorch.org/)

- [VTM](https://vcgit.hhi.fraunhofer.de/jvet/VVCSoftware_VTM)
- _This_ library (CompressAI-Vision)

Docker files including the software stack are also provided!

## For developers

After your dev, you can run (and adapt) test scripts from the scripts/tests directory. Please check scripts/tests/Readme.md for more details

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

Tutorials are produced from notebooks that are in [docs/source/tutorials](docs/source/tutorials).  If you update the notebooks, first you need to run ``compile.bash`` therein.

To produce the html documentation, from [docs/](docs/), run:
```
make html
```
To check the pages locally, open [docs/_build/html/index.html](docs/index.html)

## License

CompressAI-Vision is licensed under the BSD 3-Clause Clear License

## Authors

Fabien Racapé, Hyomin Choi, Eimran Eimon, Sampsa Riikonen, Jacky Yat-Hong Lam

## Related links
 * [HEVC HM reference software](https://hevc.hhi.fraunhofer.de)
 * [VVC VTM reference software](https://vcgit.hhi.fraunhofer.de/jvet/VVCSoftware_VTM)
 * [Detectron2](https://detectron2.readthedocs.io/en/latest/index.html)
 * [JDE](https://github.com/Zhongdao/Towards-Realtime-MOT.git)