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

"""cli register functionality
"""
import os


def add_subparser(subparsers, parents):
    subparser = subparsers.add_parser(
        "register",
        parents=parents,
        help="register image set to fiftyone from local dir",
    )
    required_group = subparser.add_argument_group("required arguments")
    required_group.add_argument(
        "--dataset-name",
        action="store",
        type=str,
        required=True,
        default=None,
        help="name of the dataset",
    )
    subparser.add_argument(
        "--lists",
        action="store",
        type=str,
        required=False,
        default=None,
        help="comma-separated list of list files",
    )
    subparser.add_argument(
        "--dir",
        action="store",
        type=str,
        required=False,
        default=None,
        help="target/source directory, depends on command",
    )
    subparser.add_argument(
        "--type",
        action="store",
        type=str,
        required=False,
        default="OpenImagesV6Dataset",
        help="image set type to be imported",
    )


def main(p):  # noqa: C901
    # fiftyone
    print("importing fiftyone")
    import fiftyone as fo

    print("fiftyone imported")

    # compressai_vision
    from compressai_vision.conversion import imageIdFileList
    from compressai_vision.tools import pathExists

    assert p.dataset_name is not None, "provide name for your dataset"
    assert p.dir is not None, "please provide path to dataset"

    try:
        dataset = fo.load_dataset(p.dataset_name)
    except ValueError:
        pass
    else:
        print("dataset %s already exists - will deregister it first" % (p.dataset_name))
        if not p.y:
            input("press enter to continue.. ")
        fo.delete_dataset(p.dataset_name)

    # if p.type != "OpenImagesV6Dataset":
    #    print("WARNING: not tested for other than OpenImagesV6Dataset - might now work")

    # dataset types are in:
    # fo.types.dataset_types.*
    # quickstart dataset is of type FiftyOneDataset
    try:
        dataset_type = getattr(fo.types.dataset_types, p.type)
    except AttributeError:
        print("WARNING: could not find dataset type from fo.types.dataset_types.*")
        print("dataset types:")
        for type_ in dir(fo.types.dataset_types):
            if type_[0] != "_":
                print(type_)
        raise

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
    print("Registering name:    ", p.dataset_name)
    if not p.y:
        input("press enter to continue.. ")
    print()
    if image_ids is None:
        dataset = fo.Dataset.from_dir(
            dataset_dir=dataset_dir,
            dataset_type=dataset_type,
            label_types=label_types,
            load_hierarchy=False,  # screw hierarchies for the moment..
            name=p.dataset_name,
        )
    else:
        dataset = fo.Dataset.from_dir(
            dataset_dir=dataset_dir,
            dataset_type=dataset_type,
            label_types=label_types,
            load_hierarchy=False,
            name=p.dataset_name,
            image_ids=image_ids,
        )
    dataset.persistent = True  # don't forget!

    print()
    print("** Let's peek at the first sample - check that it looks ok:**")
    print()
    print(dataset.first())
    print()
