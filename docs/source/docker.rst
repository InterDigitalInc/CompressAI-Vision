.. _docker:

Docker
======

Docker images are provided with the correct software stack installed into the images:

- `PyTorch <https://pytorch.org/>`_
- `CompressAI <https://interdigitalinc.github.io/CompressAI>`_
- `Detectron2 <https://detectron2.readthedocs.io/en/latest/index.html>`_
- `fiftyone <https://voxel51.com/docs/fiftyone/>`_
- *This* library (CompressAI-Vision)
- `VTM <https://vcgit.hhi.fraunhofer.de/jvet/VVCSoftware_VTM>`_

Please take a look at `docker/ <https://github.com/InterDigitalInc/CompressAI-Vision/tree/main/docker>`_, you will find:

- ``Dockerfile.*`` dockerfiles for building images
- ``make_image.bash`` helper script
- ``run_image.bash`` helper script for creating and running a container.  You should **create your own copy** of ``run_image.bash`` to suit your needs: you can use it to run an interactive notebook, create bind mounts to local directories, run batch jobs, etc.

List of docker images and software versions:

==============  ======= ========== ==== ===== ============
Dockerfile      PyTorch Detectron2 CUDA VTM   OS
==============  ======= ========== ==== ===== ============
Dockerfile.1    1.8.2   0.4        11.1 12.0  Ubuntu 20.04
==============  ======= ========== ==== ===== ============

**TODO: clean up the files & test**
