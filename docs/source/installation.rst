Installation
============

This section explains how to install the required software stack *natively* on your system.
However, you might prefer using CompressAI-Vision :ref:`via docker instead <docker>`.

1. Python dependencies
----------------------

Here we assume you are running Ubuntu 20.04 LTS (or newer)

.. _install-virtualenv:

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

After running the script within the virtualenv, you might need to ``deactivate`` and ``activate`` it for the effects to take place.


2. If your pipeline includes traditional codecs
-----------------------------------------------

Install binaries from sources, clone the selected repo from the following links and select the desired tag

- HM: <https://vcgit.hhi.fraunhofer.de/jvet/HM>
- VTM: <https://vcgit.hhi.fraunhofer.de/jvet/VVCSoftware_VTM>
- VVENC: <https://github.com/fraunhoferhhi/vvenc>

Please refer to the respective Readmes to build the binaries, that will be launched from compressai-vision code.

