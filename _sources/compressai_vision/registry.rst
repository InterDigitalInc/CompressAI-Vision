compressai_vision.registry
===========================

Register codecs, datasets, etc, to make them accessible via dynamic YAML configuration.

Registering maps a string name to a concrete creation method or class.
This allows us to dynamically create an object depending on the given string name at runtime.

.. automodule:: compressai_vision.registry


registries
-----------

.. automodule:: compressai_vision.registry.registers
   :members:
   :undoc-members: