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

"""Create a copy of a dataset for an individual user
"""
import os


def add_subparser(subparsers, parents):
    subparser = subparsers.add_parser(
        "copy", parents=parents, help="create a copy of the dataset"
    )
    req_group = subparser.add_argument_group("required arguments")
    req_group.add_argument(
        "--dataset-name",
        action="store",
        type=str,
        required=True,
        default=None,
        help="name of the dataset or a comma-separated list of dataset names",
    )
    opt_group = subparser.add_argument_group("optional arguments")
    opt_group.add_argument(
        "--username",
        action="store",
        type=str,
        required=False,
        default=None,
        help="user name to prepend the dataset names with.  default: posix username from env variable USER",
    )


def main(p):
    if p.username is None:
        p.username = os.environ["USER"]

    print("importing fiftyone")
    import fiftyone as fo

    print("fiftyone imported")

    for dataset_name in p.dataset_name.split(","):
        new_name = p.username + "-" + dataset_name
        print("cloning", dataset_name, "into", new_name)
        try:
            dataset = fo.load_dataset(dataset_name)
        except ValueError:
            print("WARNING: dataset", dataset_name, "not found - will skip")
            continue
        # delete new_name if exists already
        try:
            fo.delete_dataset(new_name)
        except ValueError:
            pass
        else:
            print("NOTE: dataset", new_name, "was already there - removed it")
        new_dataset = dataset.clone(new_name)
        new_dataset.persistent = True
