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

# compressai_vision
def main():
    # fiftyone
    import fiftyone as fo

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
    print("compressai version  :", compressai.__version__)

    print("\n*** CHECKING GPU AVAILABILITY ***")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("device              :", device)

    print("\n*** TESTING FFMPEG ***")
    c = os.system("ffmpeg -version")
    if c > 0:
        print("\nRUNNING FFMPEG FAILED\n")
    #

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
