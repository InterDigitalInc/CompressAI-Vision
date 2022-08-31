"""cli list functionality
"""
import os

# fiftyone
import fiftyone as fo

# import fiftyone.zoo as foz

# compressai_vision
# from compressai_vision.conversion import imageIdFileList


def main(p):
    print()
    print("datasets currently registered into fiftyone")
    print("name, length, first sample path")
    for name in fo.list_datasets():
        dataset = fo.load_dataset(name)
        n = len(dataset)
        if n > 0:
            sample = dataset.first()
            p = os.path.sep.join(sample["filepath"].split(os.path.sep)[:-1])
        else:
            p = "?"
        print("%s, %i, %s" % (name, len(dataset), p))
