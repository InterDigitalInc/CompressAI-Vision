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

You might want to define the python version explicitly, i.e. with ``python3.8 -m venv venv``.  Python3.8+ is required.

**While in the activated virtualenv**, run one of the ``install.bash`` bash scripts in
`scripts/ <https://github.com/InterDigitalInc/CompressAI-Vision/tree/main/scripts>`_

The script will install the following software stack:

- `PyTorch <https://pytorch.org/>`_
- `CompressAI <https://interdigitalinc.github.io/CompressAI>`_
- `Detectron2 <https://detectron2.readthedocs.io/en/latest/index.html>`_
- `fiftyone <https://voxel51.com/docs/fiftyone/>`_
- *This* library (CompressAI-Vision)

The script also inserts ``FIFTYONE_DATABASE_NAME`` env variable into the virtual environment.  This
tells fiftyone to use a database name including the particular fiftyone version (i.e. ``fiftyone-0.16.6``).

After running the script within the virtualenv, ``deactivate`` and ``activate`` the virtualenv once again for the effects to take place.

PyTorch, Detectron2 and CUDA versions are different for each of the installation scripts:

==============  ======= ========== ====
script          PyTorch Detectron2 CUDA
==============  ======= ========== ====
install.bash    1.9.1   0.6        10.2
==============  ======= ========== ====

2. VTM
------

You still need to install VTM from `here <https://vcgit.hhi.fraunhofer.de/jvet/VVCSoftware_VTM>`_.
VTM is used for anchor pipeline, i.e. for setting a baseline against which deep-learning encoders are tested.

Before compiling VTM, you need at least:

::

    sudo apt-get install build-essential wget tar cmake


Here is your quickstart for compiling VTM (adjust VTM version number accordingly):

::

    wget https://vcgit.hhi.fraunhofer.de/jvet/VVCSoftware_VTM/-/archive/VTM-12.0/VVCSoftware_VTM-VTM-12.0.tar.gz
    tar xvf VVCSoftware_VTM-VTM-12.0.tar.gz
    cd VVCSoftware_VTM-VTM-12.0
    mkdir build
    cd build
    cmake .. -DCMAKE_BUILD_TYPE=Release
    make -j
    ls ../bin

Now there should be:

::

    EncoderAppStatic
    DecoderAppStatic
    ...

Now you can try the `standalone test <https://github.com/InterDigitalInc/CompressAI-Vision/tree/main/scripts>`_

3. Other dependencies
---------------------

MPEG/VCM defines several ffmpeg commands for the anchor pipeline, so you need ffmpeg:

::

    sudo apt-get install ffmpeg

