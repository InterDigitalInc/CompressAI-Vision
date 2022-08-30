"""cli nokia_convert functionality
"""
import os

# fiftyone
import fiftyone as fo
import fiftyone.zoo as foz

# compressai_vision
from compressai_vision.conversion import imageIdFileList, nokiaBSToOpenImageV6
from compressai_vision.tools import pathExists


def main(p):
    assert p.target_dir is not None, "please give target_dir"
    assert (
        p.dir is not None
    ), "please specify OpenImageV6 source directory for images (and masks)"
    if pathExists(p.target_dir):
        print(
            "FATAL: target directory %s exists already. Please remove it first"
            % (p.target_dir)
        )
        return
    assert pathExists(p.dir), "directory " + p.dir + " does not exist"
    assert (
        p.label is not None
    ), "please provide image level labels --label ('detection_validation_labels_5k.csv')"
    image_dir = os.path.join(p.dir, "data")
    mask_dir = os.path.join(p.dir, "labels", "masks")
    assert pathExists(image_dir), "directory " + image_dir + " does not exist"
    if p.bbox is not None:
        assert pathExists(p.bbox), "file " + p.bbox + " does not exist"
        p.bbox = os.path.expanduser(p.bbox)
    if p.mask is not None:
        assert pathExists(p.mask), "file " + p.mask + " does not exist"
        assert pathExists(mask_dir), "directory " + mask_dir + " does not exist"
        p.mask = os.path.expanduser(p.mask)

    print()
    assert p.lists is not None, "a list file (.lst) required --lists"
    fnames = p.lists.split(",")
    assert len(fnames) == 1, "please specify exactly one list file for nokia convert"
    fname = fnames[0]
    print("Using list file         :    ", fname)
    print("Images (and masks) from :    ", p.dir)
    print("  --> from: ", image_dir)
    print("  --> from:", mask_dir)
    print("Image-level labels from :    ", p.label)
    print("Detections (bboxes)from :    ", p.bbox)
    print("Segmasks from           :    ", p.mask)
    print("Final OIV6 format in    :    ", p.target_dir)
    if not p.y:
        input("press enter to continue.. ")
    nokiaBSToOpenImageV6(
        validation_csv_file=os.path.expanduser(p.label),
        list_file=os.path.expanduser(fname),
        bbox_csv_file=p.bbox,
        segmentation_csv_file=p.mask,
        output_directory=os.path.expanduser(p.target_dir),
        data_dir=os.path.expanduser(image_dir),
        mask_dir=os.path.expanduser(mask_dir),
    )
    print("nokia convert ready, please check", p.target_dir)
