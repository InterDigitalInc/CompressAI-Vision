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

"""cli.py : Command-line interface tools for compressai-vision
"""
import os
import sys


def add_subparser(subparsers, parents):
    _ = subparsers.add_parser(
        "info", parents=parents, help="shows info about your system"
    )


# compressai_vision
def main(p):  # noqa: C901

    print("\n*** YOUR VIRTUALENV ***")
    print("--> running from    :", sys.executable)

    try:
        import torch
    except ModuleNotFoundError:
        print("\nPYTORCH NOT INSTALLED\n")
        sys.exit(2)
    try:
        import detectron2
    except ModuleNotFoundError:
        print("\nDETECTRON2 NOT INSTALLED\n")
        sys.exit(2)

    try:
        import compressai
    except ModuleNotFoundError:
        print("\nCOMPRESSAI NOT INSTALLED")
        sys.exit(2)

    print("\n*** TORCH, CUDA, DETECTRON2, COMPRESSAI ***")
    print("torch version       :", torch.__version__)
    print("cuda version        :", torch.version.cuda)
    print("detectron2 version  :", detectron2.__version__)
    print("--> running from    :", detectron2.__file__)
    print("compressai version  :", compressai.__version__)
    print("--> running from    :", compressai.__file__)

    print("\n*** FIFTYONE ***")
    from importlib.metadata import files, version

    util = [p for p in files("fiftyone") if "__init__.py" in str(p)][0]
    fo_path = str(util.locate())
    fo_version = version("fiftyone")
    print("fiftyone version    :", fo_version)
    print("--> running from    :", fo_path)

    print("\n*** COMPRESSAI-VISION ***")
    print("version             :", version("compressai-vision"))
    print("running from        :", __file__)

    print("\n*** CHECKING GPU AVAILABILITY ***")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("device              :", device)

    print("\n*** TESTING FFMPEG ***")
    c = os.system("ffmpeg -version")
    if c > 0:
        print("\nRUNNING FFMPEG FAILED\n")
    #
    print()
    try:
        adr = os.environ["FIFTYONE_DATABASE_URI"]
    except KeyError:
        print("NOTICE: Using mongodb managed by fiftyone")
        print("Be sure not to have extra mongod server(s) running on your system")
    else:
        print("NOTICE: You have external mongodb server configured with", adr)

    try:
        db_name = os.environ["FIFTYONE_DATABASE_NAME"]
    except KeyError:
        print(
            """
        WARNING: You should set the environment variable FIFTYONE_DATABASE_NAME
        in your virtual environment.  Different virtual environments (with different
        fiftyone versions) should NOT write to the SAME database in the same mongodb server. 
        """
        )
    else:
        print("Fiftyone database name in mongodb:", db_name)

    # fiftyone
    print("importing fiftyone..")
    import fiftyone as fo

    print("..imported")

    print("\n*** DATABASE ***")
    print("info about your connection:")

    print(fo.core.odm.database.get_db_conn())
    print()

    print("\n*** DATASETS ***")
    print("datasets currently registered into fiftyone")
    print("name, length, first sample path")
    for name in fo.list_datasets():
        dataset = fo.load_dataset(name)
        n = len(dataset)
        if n > 0:
            sample = dataset.first()
            p = os.path.sep.join(sample["filepath"].split(os.path.sep)[:-1])
        else:
            p = "?"
        print("%s, %i, %s" % (name, len(dataset), p))
    print()
