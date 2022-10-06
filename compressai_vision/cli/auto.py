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

from compressai_vision.tools import getDataFile

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
compressai-mpeg_vcm-auto-import\n
\n
parameters:

    --datadir=/path/to/datasets     directory where all datasets are downloaded by default (optional)

Automatic downloading of images and importing files provided by MPEG/VCM test conditions into fiftyone

Before running this command, put the following files into the same directory
(please note 'detection_validation_5k_bbox.csv' name inconsistency):

"""
for fname in fname_list:
    help_st += fname + "\n"
help_st += "\n"


class Namespace:
    pass


def get_(key, path=None):
    """So that we use only legit names"""
    correct_name = fname_list[fname_list.index(key)]
    if path:
        correct_name = os.path.join(path, correct_name)
    return correct_name


def get_inp(inp_, txt=""):
    """Ask user for input, provide default input"""
    input_ = input("Give " + txt + " [" + inp_ + "]: ")
    if len(input_) < 1:
        # user just pressed enter
        input_ = inp_  # use the default path
    return input_


def get_dir(dir_, txt="", make=True, check=False):
    """Ask the user for a path.  Give also a default path"""
    dir_input = input("Give " + txt + " [" + dir_ + "]: ")
    if len(dir_input) < 1:
        # user just pressed enter
        dir_input = dir_  # use the default path
    dir_input = os.path.expanduser(dir_input)
    if make:
        os.makedirs(dir_input, exist_ok=True)
    elif check:
        basepath = os.path.sep.join(
            dir_input.split(os.path.sep)[0:-1]
        )  # remove last bit of the path
        assert os.path.isdir(basepath), "path " + basepath + " does not exit"
    return dir_input


def add_subparser(subparsers, parents=[]):
    subparser = subparsers.add_parser(
        "mpeg-vcm-auto-import", parents=parents, help="auto-imports mpeg-vcm working group files, downloads necessary images from the internet, imports them to fiftyone, etc.", description=help_st
    )
    # NOTE: help is something show without using this command
    # descriptions is shown when the command is used
    subparser.add_argument(
        "--datadir",
        action="store",
        type=str,
        required=False,
        default=None,
        help="directory where all datasets are downloaded by default (optional)",
    )
    subparser.add_argument(
        "--mock", action="store_true", default=False, help="debugging switch: don't use"
    )
    subparser.add_argument(
        "--use-vcm",
        action="store_true",
        default=True,
        help="Use mpeg-vcm files bundled with compressai-vision",
    )
    subparser = subparsers.add_parser(
        "manual", parents=parents, help="shows complete manual"
    )
    # subparser = subparsers.add_parser(
    #     "info", parents=parents, help="shows info about your system", description=help_st
    # )
    # subparser = subparsers.add_parser(
    #     "mongo", parents=parents, description=help_st
    # )
    # subparser = subparsers.add_parser(
    #     "download", parents=parents, help="download an image set and register it to fiftyone.", description=help_st
    # )
    # subparser = subparsers.add_parser(
    #     "list", parents=parents, help="list all datasets registered to fiftyone", description=help_st
    # )
    # subparser = subparsers.add_parser(
    #     "register", parents=parents, help="register image set to fiftyone from local dir", description=help_st
    # )
    # subparser = subparsers.add_parser(
    #     "deregister", parents=parents, help="de-register image set from fiftyone", description=help_st
    # )
    # subparser = subparsers.add_parser(
    #     "dummy", parents=parents, help="create & register a dummy database with just the first sample", description=help_st
    # )
    # subparser = subparsers.add_parser(
    #     "clean", parents=parents, help="remove temporary databases.", description=help_st
    # )
    # # subparser = subparsers.add_parser(
    # #     "detectron2-eval", parents=parents, help="evaluate model with detectron2 using OpenImageV6", description=help_st
    # # )
    # subparser = subparsers.add_parser(
    #     "vtm", parents=parents, help="generate bitstream with the vtm video encoder", description=help_st
    # )
    # subparser = subparsers.add_parser(
    #     "convert_mpeg_to_oiv6", parents=parents, help="convert the files specified in MPEG/VCM CTC into proper OpenImageV6 format & directory structure", description=help_st
    # )


def main(p_):
    from compressai_vision.cli import convert_mpeg_to_oiv6, download, dummy, register

    dirname = p_.datadir

    for fname in fname_list:
        if p_.use_vcm:
            fname_ = getDataFile(os.path.join("mpeg_vcm_data", fname))
        else:
            fname_ = fname
        if not os.path.exists(fname_):
            print("\nFATAL: missing file", fname_)
            print(help_st)
            sys.exit(2)

    if p_.use_vcm:
        load_dir = getDataFile("mpeg_vcm_data")
    else:
        load_dir = None

    p = Namespace()
    p.mock = p_.mock
    p.y = False
    p.dataset_name = "open-images-v6"

    if dirname is None:
        dir_ = os.path.join(
            "~", "fiftyone", p.dataset_name
        )  # ~/fiftyone/open-images-v6
    else:
        dir_ = os.path.join(dirname, p.dataset_name)

    dir_ = get_dir(dir_, "path to download (MPEG/VCM subset of) OpenImageV6 ")
    source_dir = os.path.join(
        dir_, "validation"
    )  # "~/fiftyone/open-images-v6/validation"

    print("\n**DOWNLOADING**\n")
    p.lists = (
        get_("detection_validation_input_5k.lst", load_dir)
        + ","
        + get_("segmentation_validation_input_5k.lst", load_dir)
    )
    p.split = "validation"
    p.dir = dir_
    download.main(p)

    print("\n**CONVERTING MPEG/VCM DETECTION DATA TO OPENIMAGEV6 FORMAT**\n")
    p = Namespace()
    p.y = False

    if dirname is None:
        mpeg_vcm_dir = os.path.join(
            "~", "fiftyone", "mpeg-vcm-detection"
        )  # ~/fiftyone/mpeg-vcm-detection
    else:
        mpeg_vcm_dir = os.path.join(dirname, "mpeg-vcm-detection")
    mpeg_vcm_dir = get_dir(
        mpeg_vcm_dir, "imported detection dataset path", make=False, check=False
    )

    p.lists = get_("detection_validation_input_5k.lst", load_dir)
    # p.dir = "~/fiftyone/open-images-v6/validation"
    p.dir = source_dir
    # p.target_dir = "~/fiftyone/mpeg-vcm-detection"
    p.target_dir = mpeg_vcm_dir
    p.label = get_("detection_validation_labels_5k.csv", load_dir)
    p.bbox = get_("detection_validation_5k_bbox.csv", load_dir)
    p.mask = None
    convert_mpeg_to_oiv6.main(p)

    print("\n**CONVERTING MPEG/VCM SEGMENTATION DATA TO OPENIMAGEV6 FORMAT**\n")
    p = Namespace()
    p.y = False

    if dirname is None:
        mpeg_vcm_dir_seg = os.path.join(
            "~", "fiftyone", "mpeg-vcm-segmentation"
        )  # ~/fiftyone/mpeg_vcm-segmentation
    else:
        mpeg_vcm_dir_seg = os.path.join(dirname, "mpeg-vcm-segmentation")
    mpeg_vcm_dir_seg = get_dir(
        mpeg_vcm_dir_seg, "imported segmentation dataset path", make=False, check=False
    )

    p.lists = get_("segmentation_validation_input_5k.lst", load_dir)
    # p.dir = "~/fiftyone/open-images-v6/validation"
    p.dir = source_dir
    # p.target_dir = "~/fiftyone/mpeg_vcm-segmentation"
    p.target_dir = mpeg_vcm_dir_seg
    p.label = get_("segmentation_validation_labels_5k.csv", load_dir)
    p.bbox = get_("segmentation_validation_bbox_5k.csv", load_dir)
    p.mask = get_("segmentation_validation_masks_5k.csv", load_dir)
    convert_mpeg_to_oiv6.main(p)

    print("\n**REGISTERING MPEG/VCM DETECTION DATA INTO FIFTYONE**\n")
    p = Namespace()
    p.y = False
    dataset_name = get_inp("mpeg-vcm-detection", "name for detection dataset")
    p.dataset_name = dataset_name
    p.lists = get_("detection_validation_input_5k.lst", load_dir)
    p.dir = mpeg_vcm_dir
    p.type = "OpenImagesV6Dataset"
    if p_.mock:
        print("WARNING: mock/debug mode: skipping")
    else:
        register.main(p)

    print("\n**CREATING DUMMY/MOCK DETECTION DATA FOR YOUR CONVENIENCE SIR**\n")
    p = Namespace()
    p.y = False
    p.dataset_name = dataset_name
    dummy.main(p)

    print("\n**REGISTERING MPEG/VCM SEGMENTATION DATA INTO FIFTYONE**\n")
    p = Namespace()
    p.y = False
    dataset_name = get_inp("mpeg-vcm-segmentation", "name for segmentation dataset")
    p.dataset_name = dataset_name
    p.lists = get_("segmentation_validation_input_5k.lst", load_dir)
    p.dir = mpeg_vcm_dir_seg
    p.type = "OpenImagesV6Dataset"
    if p_.mock:
        print("WARNING: mock/debug mode: skipping")
    else:
        register.main(p)
    print("\nPlease continue with the compressai-vision command line tool\n")
    print("\nGOODBYE\n")


if __name__ == "__main__":
    main()
