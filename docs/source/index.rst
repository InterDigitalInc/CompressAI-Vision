.. image:: https://img.shields.io/github/license/InterDigitalInc/CompressAI-Vision?color=blue
   :target: https://github.com/InterDigitalInc/CompressAI-Vision/blob/master/LICENSE

CompressAI-Vision helps you to develop, test and evaluate compression models with standardized tests in the context of "Feature Coding for Machines" (FCM) and "Video Coding for Machines" (VCM), i.e. compression methods optimized for machine tasks algorithms such as Neural-Network (NN)-based detectors.

.. image:: ./media/images/fcvcm-scope.png

CompressAI-Vision now supports the Common Test Conditions defined by the ISO/MPEG FCM Ad-hoc Group, including standardized datasets (typically OpenImageV6 subsets), evaluation protocols (OpenImageV6) and anchor pipelines based on the compression using the state-of-the-art H.266/VCC codec.

This documentation site for FCM is in reconstruction, please also refer to the Readme files within code.

To get started, please go to through the installation steps

.. toctree::
   :hidden:

   Home <self>

.. toctree::
   :maxdepth: 1
   :caption: Getting Started
   :hidden:

   installation
   walkthrough
   cli_usage
   docker

.. toctree::
   :maxdepth: 1
   :caption: Library API
   :hidden:

   compressai_vision/codecs
   compressai_vision/evaluators
   compressai_vision/pipelines/index
   compressai_vision/model_wrappers
   compressai_vision/registry

.. toctree::
   :maxdepth: 1
   :caption: Tutorials
   :hidden:

   tutorials/index
   tutorials/convert

.. toctree::
   :maxdepth: 1
   :caption: Datasets
   :hidden:

   datasets


.. toctree::
   :caption: Development
   :hidden:

   Github repository <https://github.com/InterDigitalInc/CompressAI-Vision>
