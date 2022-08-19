Installation
============

This section explains how to install the required software stack *natively* on your system.
However, you might prefer using CompressAIVision :ref:`via docker instead <docker>`.

1. Python dependencies
----------------------

Here we assume you are running Ubuntu 20.04 LTS (or newer)

First, create and activate the virtualenv with:

.. code-block:: bash

   python3 -m venv venv
   source ./venv/bin/activate
   pip install -U pip

While in the activated virtualenv, run one of the ``install_*.bash`` bash scripts in
`bash/ <https://github.com/InterDigitalInc/CompressAI-Vision/tree/main/bash>`_

Please, take a look at one of the scripts; it will install the following software stack:

- `PyTorch <https://pytorch.org/>`_
- `CompressAI <https://interdigitalinc.github.io/CompressAI>`_
- `Detectron2 <https://detectron2.readthedocs.io/en/latest/index.html>`_
- `fiftyone <https://voxel51.com/docs/fiftyone/>`_
- *This* library (CompressAI-Vision)

PyTorch, Detectron2 and CUDA versions differ for each script:

==============  ======= ========== ====
script          PyTorch Detectron2 CUDA
==============  ======= ========== ====
install_1.bash  1.9.1   0.6        10.2
==============  ======= ========== ====

2. VTM
------

Use still need to install VTM from `here <https://vcgit.hhi.fraunhofer.de/jvet/VVCSoftware_VTM>`_. VTM is used for anchor-pipeline, i.e. for setting a baseline against 
which deep-learning encoders are tested.

