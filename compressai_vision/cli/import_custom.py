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

"""Use this stub for adding new cli commands
"""
import os

possible_types = ["sfu-hw-objects-v1", "tvd-object-tracking-v1"]


def add_subparser(subparsers, parents):
    subparser = subparsers.add_parser(
        "import-custom",
        parents=parents,
        help="import some popular custom datasets into fiftyone",
    )
    req_group = subparser.add_argument_group("required arguments")
    req_group.add_argument(
        "--dataset-type",
        action="store",
        type=str,
        required=True,
        default=None,
        help="dataset type, possible values: " + ",".join(possible_types),
    )
    req_group.add_argument(
        "--dir",
        action="store",
        type=str,
        required=True,
        default=None,
        help="root directory of the dataset",
    )


def main(p):
    p.dir = os.path.expanduser(p.dir)  # correct path in the case user uses POSIX "~"
    assert (
        p.dataset_type in possible_types
    ), "dataset-type needs to be one of these:" + str(possible_types)
    assert os.path.isdir(p.dir), "can find directory " + p.dir

    print("importing fiftyone")
    import fiftyone as fo

    # from fiftyone import ViewField as F

    print("fiftyone imported")
    try:
        dataset = fo.load_dataset(p.dataset_type)
        assert dataset is not None  # dummy
    except ValueError:
        pass
    else:
        print(
            "WARNING: dataset %s already exists: will delete and rewrite"
            % (p.dataset_type)
        )

    print()
    print("Importing a custom video format into fiftyone")
    print()
    print("Dataset type           : ", p.dataset_type)
    print("Dataset root directory : ", p.dir)
    print()
    if not p.y:
        input("press enter to continue.. ")
        print()

    # implement different (custom) datasets here
    if p.dataset_type == "sfu-hw-objects-v1":
        from compressai_vision.conversion.sfu_hw_objects_v1 import (
            register,
            video_convert,
        )
        video_convert(p.dir)
        register(p.dir)
    elif p.dataset_type == "tvd-object-tracking-v1":
        from compressai_vision.conversion.tvd_object_tracking_v1 import register
        register(p.dir)

    """maybe or maybe not?
    print("creating sfu-hw-objects-v1-dummy for your convenience, sir!")
    try:
        fo.delete_dataset("sfu-hw-objects-v1-dummy")
    except ValueError:
        pass
    # create a dummy dataset
    dummyset=dataset[ (F("name_tag") == "BasketballDrill") & (F("class_tag") == "ClassX") ]
    dummyset.clone("sfu-hw-objects-v1-dummy")
    """
