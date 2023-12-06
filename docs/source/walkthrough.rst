Walkthrough
===========

This guide provides a full walkthrough on how to use the CLI of CompressAI-Vision and configure the different parts.


.. _configuring-paths:

Configuring paths
-----------------

To evaluate a pipeline and the performances of a codec, we must tell CompressAI-Vision where to find the datasets, and where to output logs of the different runs.
Override the following paths specified in ``cfgs/paths/default.yaml``:

.. code-block:: yaml
    :caption: cfgs/paths/default.yaml

    paths:
      _common_root: "./logs"
      _run_root:     "${._common_root}/runs"


It is recommended to use paths that are somewhere outside the source code directory.

For the test data, you can referer to the directory data which contains the information to retrieve the source content and label for the considered datasets.

The paths to the root of the generated folder structure is then configured using ``dataset.config.root``, default config for datasets is as follows:
:

  .. code-block:: yaml
    :caption: cfgs/dataset/default.yaml

    type: 'DefaultDataset'
    config:
      root: '/data/dataset'
      imgs_folder: 'valid'
      annotation_file: "annotations/sample.json"
      seqinfo: "seqinfo.ini"
      dataset_name: "sample_dataset"
      ext: "png"



.. _running-evaluation:

Eval
--------

First, activate the :ref:`previously installed <install-virtualenv>` virtual environment.

To evaluate your pipeline, run
.. code-block:: bash

    # Train using conf/example.yaml, and override criterion.lmbda.
    compressai-vision-eval --config-name="eval_example" ++dataset.config.root="/data/datasets/fcm_testdata/SFU_HW_Obj/Traffic_2560x1600_30_val" dataset.config.dataset_name="Traffic_2560x1600_30_val"

.. _output-directory-structure:

Output directory structure
~~~~~~~~~~~~~~~~~~~~~~~~~~

Logs are written to the directory specified by ``"${paths._common_root}"``.
By default, this has the following directory structure:

.. code-block:: none
    :caption: ${paths._common_root} directory structure

    ${paths._common_root}/

      aim/
        main/
          .aim/

      runs/
        ...
        e4e6d4d5e5c59c69f3bd7be2/     # Aim run hash.
          checkpoints/
            runner.last.pth
          configs/
            config.yaml               # Final YAML configuration for reproducibility.
          engine/
          src/
            compressai.patch          # Auto generated git diff patch for reproducibility.
            compressai_trainer.patch  # Auto generated git diff patch for reproducibility.
          tensorboard/

Each experiment run is saved in a directory named by its run hash inside the ``runs/`` directory. This directory includes the respective model checkpoints/weights, and various configurations and diffs for better reproducibility.

The default directory structure may be reconfigured by modifying ``conf/paths/default.yaml``.



.. _aim-setup:

Viewing the experiment dashboard in Aim
---------------------------------------

This section demonstrates how to start up the Aim UI for experiment tracking. Aim allows users to compare parameters, view metrics, and visualize results.


Navigate to Aim repository
~~~~~~~~~~~~~~~~~~~~~~~~~~

Aim logs all experiments to a single directory containing an ``.aim`` repository. By default, this is located in ``./logs/aim/main``. Before running the ``aim`` commands shown later, navigate to that directory:

.. code-block:: bash

    cd "./logs/aim/main"


Local-only
~~~~~~~~~~

If the directory containing the ``.aim`` directory is directly accessible from the local machine, navigate to that directory and run:

.. code-block:: bash

    aim up --host="localhost" --port=43800

Then, open a web browser and navigate to http://localhost:43800/.


Remote host (private)
~~~~~~~~~~~~~~~~~~~~~

If the directory containing the ``.aim`` directory is on a remote host that is on an accessible private LAN, then on the remote host, navigate to that directory and run:

.. code-block:: bash

    aim up --host="0.0.0.0" --port=43800

Then, open up a web browser on the local machine and navigate to ``http://REMOTE_SERVER:PORT`` or ``http://USERNAME@REMOTE_SERVER:PORT``. The Aim UI should now be accessible.

.. note:: Anyone with access to the remote host may also be able to access the Aim UI without SSH authentication. If this is not desired, see below.


Remote host (public)
~~~~~~~~~~~~~~~~~~~~

If the directory containing the ``.aim`` directory is on a remote host that is publically accessible, then on the remote host, navigate to that directory and run:

.. code-block:: bash

    aim up --host="localhost" --port=43800

The above restricts incoming connections to those originating from the remote host itself. Then, establish local port forwarding over ssh to bind the local ``localhost:43800`` to the remote ``localhost:43800``:

.. code-block:: bash

    ssh -L "localhost:43800:localhost:43800" USERNAME@REMOTE_SERVER

Finally, open up a web browser on the local machine and navigate to http://localhost:43800/.

.. note:: Other user accounts on the remote host may also be able to bind to remote ``localhost:43800``. If this is not desired, please configure the firewall on the remote host appropriately.



.. _custom-model:

Defining a custom model
-----------------------

Ensure that the model class will be imported at runtime by adding the following to ``compressai/models/__init__.py``:

.. code-block:: python
    :caption: compressai/models/__init__.py

    from .custom import MyCustomModel

Then, create a file at ``compressai/models/custom.py``, and define and register a model as follows:

.. code-block:: python
    :caption: compressai/models/custom.py

    from compressai.registry import register_model
    from .base import CompressionModel

    @register_model("my_custom_model")
    class MyCustomModel(CompressionModel):
        def __init__(self, N, M):
            ...

Then, copy ``conf/example.yaml`` into ``conf/my_custom_model.yaml`` and customize the YAML configuration to use the custom model:

.. code-block:: yaml
    :caption: conf/my_custom_model.yaml

    model:
      name: "my_custom_model"

    hp:
      N: 128
      M: 192



.. _custom-runner:

Defining a custom Runner training loop
--------------------------------------

We provide the following pre-made runners:

- :py:class:`~compressai_trainer.runners.BaseRunner` (base compression class)
- :py:class:`~compressai_trainer.runners.ImageCompressionRunner`
- :py:class:`~compressai_trainer.runners.VideoCompressionRunner` (future release)

Begin by creating a file at ``compressai_trainer/runners/custom.py`` and then add the following line to ``compressai_trainer/runners/__init__.py``:

.. code-block:: python
    :caption: compressai_trainer/runners/__init__.py

    from .custom import CustomImageCompressionRunner

Create ``conf/runners/CustomImageCompressionRunner.yaml`` with:

.. code-block:: yaml
    :caption: conf/runners/CustomImageCompressionRunner.yaml

    type: "CustomImageCompressionRunner"

    # Additional arguments for CustomImageCompressionRunner.__init__ here:
    # some_custom_argument: "value"

Then, in ``compressai_trainer/runners/custom.py``, create a :py:class:`~catalyst.runners.runner.Runner` by inheriting from :py:class:`~compressai_trainer.runners.BaseRunner` or :py:class:`~catalyst.runners.runner.Runner`:

.. code-block:: python
    :caption: compressai_trainer/runners/custom.py

    from compressai.registry import register_runner
    from .base import BaseRunner

    @register_runner("CustomImageCompressionRunner")
    class CustomImageCompressionRunner(BaseRunner):
        ...

The following functions are called during the training loop:

.. code-block:: python
    :caption: Runner training loop call order.

    on_experiment_start   # Once, at the beginning.
      on_epoch_start      # Beginning of an epoch.
        on_loader_start   # For each loader (train / valid / infer).
          on_batch_start  # Before each batch.
            handle_batch  # For each image batch.
          on_batch_end
        on_loader_end
      on_epoch_end
    on_experiment_end

The training loop is effectively equivalent to:

.. code-block:: python
    :caption: Runner training loop pseudo-code.

    on_experiment_start()

    for epoch in range(1, num_epochs):
        on_epoch_start()

        for loader in ["train", "valid", "infer"]:
            on_loader_start()

            for batch in loader:
                on_batch_start()
                handle_batch(batch)
                on_batch_end()

            on_loader_end()

        on_epoch_end()

    on_experiment_end()

Please see the `Catalyst documentation`_ for more information. Also consider using our provided runners as a template.

.. _Catalyst documentation: https://catalyst-team.github.io/catalyst/



.. _yaml-config:

Using YAML configuration
------------------------

We use Hydra for our configuration framework. The section below covers some basics, but for more details, please see the `Hydra documentation`_.

.. _Hydra documentation: https://hydra.cc/docs/intro/


Basics
~~~~~~

``conf/example.yaml`` contains an example configuration for training the ``bmshj2018-factorized`` model.

In the ``defaults`` list, one may import configurations from other YAML files:


.. code-block:: yaml
    :caption: conf/example.yaml

    defaults:
      # Imports conf/runner/ImageCompressionRunner.yaml into "runner:" dict.
      - runner: ImageCompressionRunner

      # Similarly, import into "paths:", "env:", "engine:", etc dicts.
      - paths: default
      - env: default
      - engine: default
      - criterion: RateDistortionLoss
      - optimizer: net_aux
      - scheduler: ReduceLROnPlateau
      - misc: default

      # Imports vimeo90k/train into "dataset.train:" dict, etc.
      - dataset@dataset.train: vimeo90k/train
      - dataset@dataset.valid: vimeo90k/valid
      - dataset@dataset.infer: kodak/infer

      # Imports current YAML's configuration, defined below.
      - _self_

One may also define or override configuration within the YAML file itself:

.. code-block:: yaml
    :caption: conf/example.yaml

    # Create configuration for current experiment, model, and hyperparameters.
    exp:
      name: "example_experiment"
    model:
      name: "bmshj2018-factorized"
    hp:
      N: 128
      M: 192

    # Override dataset.train.loader.batch size.
    dataset:
      train:
        loader:
          batch_size: 8

    # Alternatively, one can also override the above via a command line argument:
    # compressai-train [...] ++dataset.train.loader.batch_size=8


Creating your own config
~~~~~~~~~~~~~~~~~~~~~~~~

Copy ``conf/example.yaml`` into ``conf/custom-config.yaml``:

.. code-block:: yaml
    :caption: conf/custom-config.yaml

    defaults:
      - paths: default
      - env: default
      - engine: default
      - runner: ImageCompressionRunner
      - criterion: RateDistortionLoss
      - optimizer: net_aux
      - scheduler: ReduceLROnPlateau
      - dataset/vimeo90k/train@dataset
      - dataset/vimeo90k/valid@dataset
      - dataset/kodak/infer@dataset
      - misc: default
      - _self_

    exp:
      name: "example_experiment"

    model:
      name: "bmshj2018-factorized"

    hp:
      # Qualities 1-5
      N: 128
      M: 192
      # Qualities 6-8
      # N: 192
      # M: 320

Modify it as desired. Then, train using:

.. code-block:: bash

    compressai-train --config-name="custom-config"


Specify and override configuration via command line (CLI) arguments
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For example, this overrides ``criterion.lmbda``:

.. code-block:: bash

    compressai-train --config-name="example" ++criterion.lmbda=0.035

The above is equivalent to the following YAML configuration:

.. code-block:: yaml

    criterion:
      lmbda: 0.035

Please see the `Hydra documentation on overriding`_ for more information.

.. _Hydra documentation on overriding: https://hydra.cc/docs/advanced/override_grammar/basic/#basic-examples



.. _resuming-training:

Resuming training
-----------------

Loading model checkpoints/weights
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. note:: This starts a *fresh* run in the experiment tracker with a new run hash. To log to an existing run, see :ref:`continuing-run`.

To load a checkpoint containing model/optimizer/scheduler/etc state, override ``paths.checkpoint``:

.. code-block:: bash

    ++paths.checkpoint="/path/to/checkpoints/runner.last.pth"

To load *only* the model ``state_dict`` (i.e. weights), and not other training state, override ``paths.model_checkpoint`` instead:

.. code-block:: bash

    ++paths.model_checkpoint="/path/to/checkpoints/runner.last.pth"


.. _continuing-run:

Continuing a previous run
~~~~~~~~~~~~~~~~~~~~~~~~~

To continue an existing run that was paused/cancelled during training, load the config corresponding to the run hash:

.. code-block:: bash

    RUNS_ROOT="${paths.runs_root}"  # Example: "./logs/runs"
    RUN_HASH="${env.aim.run_hash}"  # Example: "e4e6d4d5e5c59c69f3bd7be2"

    --config-path="${RUNS_ROOT}/${RUN_HASH}/configs"
    --config-name="config"
    ++paths.checkpoint='${paths.checkpoints}/runner.last.pth'



.. _additional-loggers:

Additional loggers
------------------

By default, CompressAI Trainer logs experiments to both Aim and Tensorboard. Additional loggers can be enabled as shown below.

CSV Logger
~~~~~~~~~~

Store CSV logs inside the current run directory via:

.. code-block:: bash

    compressai-train \
      --config-name="example" \
      ++engine.loggers.csv.logdir='${paths._run_root}/csv'


MLflow Logger
~~~~~~~~~~~~~

Connect CompressAI Trainer to an MLflow experiment tracking server:

.. code-block:: bash

    compressai-train \
      --config-name="example" \
      ++exp.name="example_experiment" \
      ++engine.loggers.mlflow.run="example_run" \
      ++engine.loggers.mlflow.tracking_uri=http://localhost:5000 \
      ++engine.loggers.mlflow.registry_uri=http://localhost:5000



.. _tips:

Tips
----

Single-GPU and Multi-GPU training
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

By default, CompressAI Trainer will use all available GPUs. To restrict to certain GPU devices, set the visible device IDs using:

.. code-block:: bash

    export CUDA_VISIBLE_DEVICES="0,1"  # Restricts to GPU 0 and GPU 1.


Quick sanity check
~~~~~~~~~~~~~~~~~~

To quickly check that your code is working, run a few batches of train/validation/inference using the following CLI argument:

.. code-block:: bash

    ++engine.check=True

To avoid filling up the ``"${paths.runs_root}"`` directory with unnecessary checkpoints, we recommend adding the following variable to ``~/.bashrc``:

.. code-block:: bash

    TRAIN_CHECK="++engine.check=True ++exp.name=check ++paths.runs_root=$USER/tmp_runs"

Example usage:

.. code-block:: bash

    compressai-train --config-name="example" $TRAIN_CHECK

