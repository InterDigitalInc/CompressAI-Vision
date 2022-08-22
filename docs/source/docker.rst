.. _docker:

Docker
======

**TODO: dockerfiles try to pull CompressAI-Vision from the public repo: this does not work since the repo is not yet public**

Docker images are provided with the correct software stack:

- `PyTorch <https://pytorch.org/>`_
- `CompressAI <https://interdigitalinc.github.io/CompressAI>`_
- `Detectron2 <https://detectron2.readthedocs.io/en/latest/index.html>`_
- `fiftyone <https://voxel51.com/docs/fiftyone/>`_
- *This* library (CompressAI-Vision)
- `VTM <https://vcgit.hhi.fraunhofer.de/jvet/VVCSoftware_VTM>`_

Please take a look at `docker/ <https://github.com/InterDigitalInc/CompressAI-Vision/tree/main/docker>`_, you will find:

- ``docker-driver.bash`` : run this file once in order to be able to use nvidia GPUs with docker
- ``Dockerfile.*`` dockerfiles for building images
- ``make_image.bash`` helper script
- ``run_image.bash`` helper script for creating and running a container. 

To create image ``compressai_vision:1`` using ``Dockerfile.1`` just use:

::

    ./make_image.bash 1

You should **create your own copy** of ``run_image.bash`` that suits your needs: you can use it to run an interactive notebook, create bind mounts to local directories, run batch jobs, etc.

The default demo version starts a python notebook and tensorboard:

::

    ./run_image.bash 1 gpu

File versions
=============

List of docker images and software versions:

==============  ======= ========== ==== ===== ============ ===================================================================
Dockerfile      PyTorch Detectron2 CUDA VTM   OS           Description
==============  ======= ========== ==== ===== ============ ===================================================================
Dockerfile.1    1.8.2   0.4        11.1 12.0  Ubuntu 20.04 All inclusive image,
                                                           run with ``run_image.bash``
Dockerfile.2    1.8.2   0.4        11.1 12.0  Ubuntu 20.04 Hot-reload CompressAI-Vision (for compressai-vision development),
                                                           run with ``run_image_dev.bash``
==============  ======= ========== ==== ===== ============ ===================================================================

