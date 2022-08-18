"""cli.py : Command-line interface tools for compressai-vision

* Copyright: 2022 InterDigital
* Authors  : Sampsa Riikonen
* Date     : 2022
* Version  : 0.1

This file is part of compressai-vision libraru
"""
import os, sys

# define legit filenames.. there are inconsistencies here
fname_list=["detection_validation_input_5k.lst",
    "detection_validation_5k_bbox.csv", # inconsistent name
    "detection_validation_labels_5k.csv",
    "segmentation_validation_input_5k.lst",
    "segmentation_validation_bbox_5k.csv",
    "segmentation_validation_labels_5k.csv",
    "segmentation_validation_masks_5k.csv"]

help_st="""
compressai-nokia-auto-import

Automatic downloading of images and importing nokia-provided files into fiftyone

Before running this command (without any arguments), put the following files into the same directory 
(please note 'detection_validation_5k_bbox.csv' name inconsistency):

"""
for fname in fname_list:
    help_st += fname + "\n"
help_st += "\n"

class Namespace:
    pass

def get_(key):
    """So that we use only legit names
    """
    return fname_list[fname_list.index(key)]

def main():
    from compressai_vision.cli import download, nokia_convert, register

    if len(sys.argv) > 1:
        print(help_st)
        return

    for fname in fname_list:
        if not os.path.exists(fname):
            print("\nFATAL: missing file", fname)
            print(help_st)
            sys.exit(2)

    print("\n**DOWNLOADING**\n")
    p=Namespace()
    p.y=False
    p.name="open-images-v6"
    p.lists=get_("detection_validation_input_5k.lst")+","+get_("segmentation_validation_input_5k.lst")
    p.split="validation"
    download(p)

    print("\n**CONVERTING DETECTION DATA**\n")
    p=Namespace()
    p.y=False
    p.lists=get_("detection_validation_input_5k.lst")
    p.dir="~/fiftyone/open-images-v6/validation"
    p.target_dir="~/fiftyone/nokia-detection"
    p.label=get_("detection_validation_labels_5k.csv")
    p.bbox=get_("detection_validation_5k_bbox.csv")
    p.mask=None
    nokia_convert(p)

    print("\n**CONVERTING SEGMENTATION DATA**\n")
    p=Namespace()
    p.y=False
    p.lists=get_("segmentation_validation_input_5k.lst")
    p.dir="~/fiftyone/open-images-v6/validation"
    p.target_dir="~/fiftyone/nokia-segmentation"
    p.label=get_("segmentation_validation_labels_5k.csv")
    p.bbox=get_("segmentation_validation_bbox_5k.csv")
    p.mask=get_("segmentation_validation_masks_5k.csv")
    nokia_convert(p)

    print("\n**REGISTERING DETECTION DATA**\n")
    p=Namespace()
    p.y=False
    p.name="nokia-detection"
    p.lists=get_("detection_validation_input_5k.lst")
    p.dir="~/fiftyone/nokia-detection"
    p.type="OpenImagesV6Dataset"
    register(p)

    print("\n**REGISTERING SEGMENTATION DATA**\n")
    p=Namespace()
    p.y=False
    p.name="nokia-segmentation"
    p.lists=get_("segmentation_validation_input_5k.lst")
    p.dir="~/fiftyone/nokia-segmentation"
    p.type="OpenImagesV6Dataset"
    register(p)
    print("\nPlease continue with the compressai-vision command line tool\n")
    print("\nGOODBYE\n")
    

if (__name__ == "__main__"):
    main()
