"""cli create dummy db functionality
"""
import os

# fiftyone
import fiftyone as fo

# import fiftyone.zoo as foz
# from compressai_vision.conversion import imageIdFileList


def main(p):
    try:
        dataset = fo.load_dataset(p.name)
    except ValueError:
        print("dataset", p.name, "does not exist!")
        return
    dummyname = p.name + "-dummy"
    print("creating dataset", dummyname)
    try:
        fo.delete_dataset(dummyname)
    except ValueError:
        pass
    dummy_dataset = fo.Dataset(dummyname)
    for sample in dataset[0:1]:
        dummy_dataset.add_sample(sample)
    dummy_dataset.persistent = True
