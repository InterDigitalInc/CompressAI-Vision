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

"""
# importing this takes quite a while!
# ..not anymore since import fiftyone is inside the function
# print("cli: import")
from compressai_vision.cli.clean import main as clean
from compressai_vision.cli.convert_mpeg_to_oiv6 import main as convert_mpeg_to_oiv6
from compressai_vision.cli.deregister import main as deregister
from compressai_vision.cli.detectron2_eval import main as detectron2_eval
from compressai_vision.cli.download import main as download
from compressai_vision.cli.dummy import main as dummy
from compressai_vision.cli.list import main as list
from compressai_vision.cli.load_eval import main as load_eval
from compressai_vision.cli.register import main as register
from compressai_vision.cli.vtm import main as vtm

# print("cli: import end")
"""
from . import clean, convert_mpeg_to_oiv6, deregister, detectron2_eval,\
    download, dummy, list_, load_eval, register, vtm, auto, info

__all__ = [
 "clean", "convert_mpeg_to_oiv6", "deregister", "detectron2_eval",
    "download", "dummy", "list_", "load_eval", "register", "vtm", "auto", "info"
]
