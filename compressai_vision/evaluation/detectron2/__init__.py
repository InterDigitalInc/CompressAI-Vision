"""The old pipeline where we tried to use the Detectron2 library only for
database handling, evaluation, etc.
"""
raise AssertionError("legacy code")
from .inference import inference_on_dataset
from .mapper import EncodingDecodingDatasetMapper
from .tools import filterInstances, mapDataset, mapInputDict, mapInstances
