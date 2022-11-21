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

"""monkey-patching for https://github.com/voxel51/fiftyone/issues/2096
"""
# import importhook # this module simply ...ks up everything (at least torch imports)
from datetime import datetime
from importlib.metadata import version
import fiftyone as fo  # importing fiftyone for the first time always takes time as it starts the mongodb


if version("fiftyone") != "0.16.6":
    print("")
    print("*** WARNING *** WARNING *** WARNING *** WARNING ***")
    print(
        "PATCHING FOR FIFTYONE VERSION 0.16.6 BUT YOUR VERSION IS", version("fiftyone")
    )
    print("")

"""as per
https://github.com/voxel51/fiftyone/blob/develop/fiftyone/core/dataset.py#L5616
https://github.com/voxel51/fiftyone/issues/2096

relevant when sending parallel queue jobs
"""


def _make_sample_collection_name(patches=False, frames=False, clips=False):
    # conn = foo.get_db_conn()
    now = datetime.now()

    if patches:
        prefix = "patches"
    elif frames:
        prefix = "frames"
    elif clips:
        prefix = "clips"
    else:
        prefix = "samples"

    # TODO (sampsa) E731 do not assign a lambda expression, use a def:
    create_name = lambda timestamp: ".".join([prefix, timestamp])  # noqa: E731

    name = create_name(now.strftime("%Y.%m.%d.%H.%M.%S.%f"))
    # print(">PATCH") # checking that this method is picked up..
    return name


"""as per 
https://github.com/voxel51/fiftyone/issues/2291

relevant when importing non-canonical OpenImageV6 formats into fiftyone
"""
import csv
import pandas as pd


def _parse_csv(filename, dataframe=False, index_col=None):
    if dataframe:
        data = pd.read_csv(filename, index_col=index_col, dtype={"ImageID": str})
    else:
        with open(filename, "r", newline="", encoding="utf8") as csvfile:
            dialect = csv.Sniffer().sniff(csvfile.read(10240))
            csvfile.seek(0)
            if dialect.delimiter in fo.utils.openimages._CSV_DELIMITERS:
                reader = csv.reader(csvfile, dialect)
            else:
                reader = csv.reader(csvfile)

            data = [row for row in reader]

    return data


# apply patches
fo.core.dataset._make_sample_collection_name = _make_sample_collection_name
import fiftyone.utils.openimages as fouo

fouo._parse_csv = _parse_csv
