# Copyright (c) 2022, InterDigital Communications, Inc
# All rights reserved.

# Redistribution and use in source and binary forms, with or without
# modification, are permitted (subject to the limitations in the disclaimer
# below) provided that the following conditions are met:

# * Redistributions of source code must retain the above copyright notice,
#   this list of conditions and the following disclaimer.
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
# * Neither the name of InterDigital Communications, Inc nor the names of its
#   contributors may be used to endorse or promote products derived from this
#   software without specific prior written permission.

# NO EXPRESS OR IMPLIED LICENSES TO ANY PARTY'S PATENT RIGHTS ARE GRANTED BY
# THIS LICENSE. THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND
# CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT
# NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
# PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS;
# OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
# WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR
# OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF
# ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

"""cli : Command-line interface tools for compressai-vision
"""
import os
import sys

# define legit filenames.. there are inconsistencies here
fname_list = [
    "detection_validation_input_5k.lst",
    "detection_validation_5k_bbox.csv",  # inconsistent name
    "detection_validation_labels_5k.csv",
    "segmentation_validation_input_5k.lst",
    "segmentation_validation_bbox_5k.csv",
    "segmentation_validation_labels_5k.csv",
    "segmentation_validation_masks_5k.csv",
]

help_st = """
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
    """So that we use only legit names"""
    return fname_list[fname_list.index(key)]


def main():
    from compressai_vision.cli import download, nokia_convert, register, dummy

    if len(sys.argv) > 1:
        print(help_st)
        return

    for fname in fname_list:
        if not os.path.exists(fname):
            print("\nFATAL: missing file", fname)
            print(help_st)
            sys.exit(2)

    print("\n**DOWNLOADING**\n")
    p = Namespace()
    p.y = False
    p.name = "open-images-v6"
    p.lists = (
        get_("detection_validation_input_5k.lst")
        + ","
        + get_("segmentation_validation_input_5k.lst")
    )
    p.split = "validation"
    download(p)

    print("\n**CONVERTING DETECTION DATA**\n")
    p = Namespace()
    p.y = False
    p.lists = get_("detection_validation_input_5k.lst")
    p.dir = "~/fiftyone/open-images-v6/validation"
    p.target_dir = "~/fiftyone/nokia-detection"
    p.label = get_("detection_validation_labels_5k.csv")
    p.bbox = get_("detection_validation_5k_bbox.csv")
    p.mask = None
    nokia_convert(p)

    print("\n**CONVERTING SEGMENTATION DATA**\n")
    p = Namespace()
    p.y = False
    p.lists = get_("segmentation_validation_input_5k.lst")
    p.dir = "~/fiftyone/open-images-v6/validation"
    p.target_dir = "~/fiftyone/nokia-segmentation"
    p.label = get_("segmentation_validation_labels_5k.csv")
    p.bbox = get_("segmentation_validation_bbox_5k.csv")
    p.mask = get_("segmentation_validation_masks_5k.csv")
    nokia_convert(p)

    print("\n**REGISTERING DETECTION DATA**\n")
    p = Namespace()
    p.y = False
    p.name = "nokia-detection"
    p.lists = get_("detection_validation_input_5k.lst")
    p.dir = "~/fiftyone/nokia-detection"
    p.type = "OpenImagesV6Dataset"
    register(p)

    print("\n**CREATING DUMMY/MOCK DETECTION DATA FOR YOUR CONVENIENCE SIR**\n")
    p = Namespace()
    p.y = False
    p.name = "nokia-detection"
    dummy(p)

    print("\n**REGISTERING SEGMENTATION DATA**\n")
    p = Namespace()
    p.y = False
    p.name = "nokia-segmentation"
    p.lists = get_("segmentation_validation_input_5k.lst")
    p.dir = "~/fiftyone/nokia-segmentation"
    p.type = "OpenImagesV6Dataset"
    register(p)
    print("\nPlease continue with the compressai-vision command line tool\n")
    print("\nGOODBYE\n")


if __name__ == "__main__":
    main()
