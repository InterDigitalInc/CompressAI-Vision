Installation
============

This section explains how to install the required software stack *natively* on your system.
However, you might prefer using CompressAI-Vision :ref:`via docker instead <docker>`.

1. Python dependencies
----------------------

Here we assume you are running Ubuntu 20.04 LTS (or newer)

First, create and activate the virtualenv with:

.. code-block:: bash

   python3 -m venv venv
   source ./venv/bin/activate
   pip install -U pip

You might want to define the python version explicitly, i.e. with ``python3.8 -m venv venv``.  Python3.8+ is required.

**While in the activated virtualenv**, run the installation:

.. code-block:: bash

   bash scripts/install.sh

The script will install the following software stack:

- `PyTorch <https://pytorch.org/>`_
- `Detectron2 <https://detectron2.readthedocs.io/en/latest/index.html>`_
- *This* library (CompressAI-Vision)

After running the script within the virtualenv, ``deactivate`` and ``activate`` the virtualenv once again for the effects to take place.

PyTorch, Detectron2 and CUDA versions are different for each of the installation scripts:

==============  ======= ========== ====
script          PyTorch Detectron2 CUDA
==============  ======= ========== ====
install.bash    2.0.0   0.6        11.8
==============  ======= ========== ====

2. If your pipeline includes traditional codecs
------

Install binaries from sources, clone the selected repo from the following links and select the desired tag
- HM
- VTM: <https://vcgit.hhi.fraunhofer.de/jvet/VVCSoftware_VTM>
- VVENC

Please refer to the respective Readmes to build the binaries, that will be launched from compressai-vision code.

