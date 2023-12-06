.. _docker:

Docker
======

*WARNING: as for the moment, docker images are not supported / up-to-date*

Docker images are provided with the correct software stack:

For FCM pipelines:
- `PyTorch <https://pytorch.org/>`_
- `Detectron2 <https://detectron2.readthedocs.io/en/latest/index.html>`_
- `JDE <https://github.com/Zhongdao/Towards-Realtime-MOT>`_

For VCM pipelines:
- `PyTorch <https://pytorch.org/>`_
- `CompressAI <https://interdigitalinc.github.io/CompressAI>`_
- `Detectron2 <https://detectron2.readthedocs.io/en/latest/index.html>`_
- `fiftyone <https://voxel51.com/docs/fiftyone/>`_
- *This* library (CompressAI-Vision)
- `VTM <https://vcgit.hhi.fraunhofer.de/jvet/VVCSoftware_VTM>`_

Please take a look at `docker/ <https://github.com/InterDigitalInc/CompressAI-Vision/tree/main/docker>`_, you will find for each FCM/VCM case:

- ``docker-driver.bash`` : run this file once in order to be able to use nvidia GPUs with docker
- ``Dockerfile.*`` dockerfiles for building images
- ``make_image.bash`` helper script
- ``run_image.bash`` helper script for creating and running a container.

To create image ``compressai_vision:vcm`` using ``Dockerfile`` just use:

::

    ./make_image.bash 1

You should **create your own copy** of ``run_image.bash`` that suits your needs: you can use it to run an interactive notebook, create bind mounts to local directories, run batch jobs, etc.

The default demo version starts a python notebook and tensorboard:

::

    ./run_image.bash 1 gpu

File versions
-------------

List of docker images and software versions:

==============  ======= ========== ==== ===== ====== =============== ===================================================================
Dockerfile      PyTorch Detectron2 CUDA VTM   FFMpeg OS              Description
==============  ======= ========== ==== ===== ====== =============== ===================================================================
Dockerfile.FCM  2.0.0   1.0        11.3 12.0  4.2.7  Ubuntu 20.04    All inclusive image,
                                                                     run with ``run_image.bash``
Dockerfile.VCM  1.8.2   0.4        11.1 12.0  4.2.7  Ubuntu 20.04    Hot-reload CompressAI-Vision (for compressai-vision development),
                                                                     run with ``run_image.bash``.
==============  ======= ========== ==== ===== ====== =============== ===================================================================
