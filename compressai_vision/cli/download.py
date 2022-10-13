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

"""cli download functionality
"""
import os


def add_subparser(subparsers, parents):
    subparser = subparsers.add_parser(
        "download",
        parents=parents,
        help="download an image set and register it to fiftyone.",
    )
    required_group = subparser.add_argument_group("required arguments")
    subparser.add_argument(
        "--mock", action="store_true", default=False, help="mock tests"
    )
    required_group.add_argument(
        "--dataset-name",
        action="store",
        type=str,
        required=True,
        default="open-images-v6",
        help="name of the dataset",
    )
    subparser.add_argument(
        "--lists",
        action="store",
        type=str,
        required=False,
        default=None,
        help="comma-separated list of list files. Example: detection_validation_input_5k.lst, segmentation_validation_input_5k.lst",
    )
    subparser.add_argument(
        "--split",
        action="store",
        type=str,
        required=False,
        default=None,
        help="database sub-name. Example: 'train' or 'validation'",
    )
    subparser.add_argument(
        "--dir",
        action="store",
        type=str,
        required=False,
        default=None,
        help="directory where the dataset (images, annotations, etc.) is downloaded. default uses fiftyone default, i.e. ~/fiftyone/",
    )


def main(p):
    # fiftyone
    print("importing fiftyone")
    import fiftyone as fo

    from fiftyone import zoo as foz  # different fiftyone than the patched one.. eh

    print("fiftyone imported")

    # compressai_vision
    from compressai_vision.conversion import imageIdFileList
    from compressai_vision.tools import pathExists

    if p.dataset_name is None:
        p.dataset_name = "open-images-v6"
    print()
    if p.lists is None:
        print(
            "WARNING: downloading ALL images.  You might want to use the --lists option to download only certain images"
        )
        n_images = "?"
        image_ids = None
    else:
        fnames = p.lists.split(",")
        for fname in fnames:
            assert pathExists(fname)
        image_ids = imageIdFileList(*fnames)
        if p.mock:
            image_ids = image_ids[0:2]
            print("WARNING! MOCK TEST OF ONLY TWO SAMPLES!")
        n_images = str(len(image_ids))
    print("Using list files:    ", p.lists)
    print("Number of images:    ", n_images)
    print("Database name   :    ", p.dataset_name)
    print("Subname/split   :    ", p.split)
    print("Target dir      :    ", p.dir)
    if not p.y:
        input("press enter to continue.. ")
    print()
    # return
    # https://voxel51.com/docs/fiftyone/user_guide/dataset_zoo/datasets.html#dataset-zoo-open-images-v6
    kwargs = {}
    if p.split is not None:
        kwargs["split"] = p.split
    if image_ids is not None:
        kwargs["image_ids"] = image_ids
    if p.dir is not None:
        p.dir = os.path.expanduser(p.dir)
        kwargs["dataset_dir"] = p.dir
    # print(">>>", p.dir)
    dataset = foz.load_zoo_dataset(p.dataset_name, **kwargs)
    # label_types=("detections", "classifications", "relationships", "segmentations") # default
    dataset.persistent = True
