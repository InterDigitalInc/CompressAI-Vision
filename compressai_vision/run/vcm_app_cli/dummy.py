# Copyright (c) 2022-2024 InterDigital Communications, Inc
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

"""cli create dummy db functionality
"""


def add_subparser(subparsers, parents):
    subparser = subparsers.add_parser(
        "dummy",
        parents=parents,
        help="create & register a dummy database with just the first sample",
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


def main(p):
    # fiftyone
    print("importing fiftyone")
    import fiftyone as fo

    print("fiftyone imported")

    try:
        dataset = fo.load_dataset(p.dataset_name)
    except ValueError:
        print("dataset", p.dataset_name, "does not exist!")
        return
    dummyname = p.dataset_name + "-dummy"
    print("creating dataset", dummyname)
    try:
        fo.delete_dataset(dummyname)
    except ValueError:
        pass
    dummy_dataset = fo.Dataset(dummyname)
    for sample in dataset[0:1]:
        dummy_dataset.add_sample(sample)
    dummy_dataset.persistent = True
