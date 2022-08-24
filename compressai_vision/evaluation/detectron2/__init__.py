"""The old pipeline where we tried to use the Detectron2 library only for database handling, evaluation, etc.
"""
raise AssertionError("legacy code")
from .inference import inference_on_dataset
from .tools import mapInputDict, mapInstances, mapDataset,\
    filterInstances
from .mapper import EncodingDecodingDatasetMapper
