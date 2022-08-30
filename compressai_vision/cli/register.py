"""cli register functionality
"""
import os

# fiftyone
import fiftyone as fo
import fiftyone.zoo as foz

# compressai_vision
from compressai_vision.conversion import imageIdFileList
from compressai_vision.tools import pathExists


def main(p):
    assert p.name is not None, "provide name for your dataset"
    assert p.dir is not None, "please provide path to dataset"

    try:
        dataset = fo.load_dataset(p.name)
    except ValueError as e:
        pass
    else:
        print("dataset %s already exists - will deregister it first" % (p.name))
        if not p.y:
            input("press enter to continue.. ")
        fo.delete_dataset(p.name)

    if p.type != "OpenImagesV6Dataset":
        print("WARNING: not tested for other than OpenImagesV6Dataset - might now work")
    dataset_type = getattr(fo.types, p.type)
    dataset_dir = os.path.expanduser(p.dir)
    assert pathExists(dataset_dir)
    print()
    if p.lists is None:
        print(
            "WARNING: using/registering with ALL images.  You should use the --lists option"
        )
        n_images = "?"
        image_ids = None
    else:
        fnames = p.lists.split(",")
        for fname in fnames:
            assert pathExists(fname), "file " + fname + " does not exist"
        image_ids = imageIdFileList(*fnames)
        n_images = str(len(image_ids))

    label_types = [
        "classifications"
    ]  # at least image-level classifications required..!
    # let's check what data user has imported
    if pathExists(os.path.join(p.dir, "labels", "segmentations.csv")):
        # segmentations are there allright
        label_types.append("segmentations")
    if pathExists(os.path.join(p.dir, "labels", "detections.csv")):
        # segmentations are there allright
        label_types.append("detections")
    # .. in fact, could just list with all .csv files in that dir

    print("From directory  :    ", p.dir)
    print("Using list file :    ", p.lists)
    print("Number of images:    ", n_images)
    print("Registering name:    ", p.name)
    if not p.y:
        input("press enter to continue.. ")
    print()
    if image_ids is None:
        dataset = fo.Dataset.from_dir(
            dataset_dir=dataset_dir,
            dataset_type=dataset_type,
            label_types=label_types,
            load_hierarchy=False,  # screw hierarchies for the moment..
            name=p.name,
        )
    else:
        dataset = fo.Dataset.from_dir(
            dataset_dir=dataset_dir,
            dataset_type=dataset_type,
            label_types=label_types,
            load_hierarchy=False,
            name=p.name,
            image_ids=image_ids,
        )
