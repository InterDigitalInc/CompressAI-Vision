"""cli download functionality
"""
# import os

# fiftyone
# import fiftyone as fo
import fiftyone.zoo as foz

# compressai_vision
from compressai_vision.conversion import imageIdFileList
from compressai_vision.tools import pathExists


def main(p):
    if p.name is None:
        p.name = "open-images-v6"
    print()
    if p.lists is None:
        print(
            "WARNING: downloading ALL images.  You should use the --lists option instead."
        )
        n_images = "?"
        image_ids = None
    else:
        fnames = p.lists.split(",")
        for fname in fnames:
            assert pathExists(fname)
        image_ids = imageIdFileList(*fnames)
        n_images = str(len(image_ids))
    print("Using list files:    ", p.lists)
    print("Number of images:    ", n_images)
    print("Database name   :    ", p.name)
    print("Subname/split   :    ", p.split)
    if not p.y:
        input("press enter to continue.. ")
    print()
    # return
    # https://voxel51.com/docs/fiftyone/user_guide/dataset_zoo/datasets.html#dataset-zoo-open-images-v6
    if image_ids is None:
        dataset = foz.load_zoo_dataset(
            p.name,
            split=p.split,
            # label_types=("detections", "classifications", "relationships", "segmentations") # default
        )
    else:
        dataset = foz.load_zoo_dataset(p.name, split=p.split, image_ids=image_ids)
    dataset.persistent = True
