# Copyright (c) 2022-2024, InterDigital Communications, Inc
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


import hashlib
import zipfile

from collections import OrderedDict
from collections.abc import Mapping, Sequence
from contextlib import contextmanager
from typing import Tuple

import torch


class FileLikeHasher:
    def __init__(self, fn, algo: str = "md5"):
        self._h = hashlib.new(algo)
        self._fn = fn
        self._nbytes = 0

    def write(self, byts):
        self._h.update(byts)
        self._nbytes += len(byts)
        return len(byts)

    def flush(self):
        pass

    def close(self):
        with open(self._fn, "w") as f:
            f.write(self._h.hexdigest())
            f.write("\n")


@contextmanager
def freeze_zip_timestamps(
    fixed: Tuple[int, int, int, int, int, int] = (1980, 1, 1, 0, 0, 0),
):
    _orig_init = zipfile.ZipInfo.__init__

    def _patched(self, *args, **kwargs):
        _orig_init(self, *args, **kwargs)
        self.date_time = fixed  # ZIP fixed time

    zipfile.ZipInfo.__init__ = _patched
    try:
        yield
    finally:
        zipfile.ZipInfo.__init__ = _orig_init


def contiguous_features(obj):
    if isinstance(obj, torch.Tensor):
        return obj.to("cpu").contiguous().clone()

    if isinstance(obj, Mapping):
        return OrderedDict(
            (k, contiguous_features(v))
            for k, v in sorted(obj.items(), key=lambda item: str(item[0]))
            if not str(k).startswith("file")
        )

    if isinstance(obj, set):
        return tuple(sorted(obj, key=str))

    if isinstance(obj, Sequence) and not isinstance(obj, (str, bytes)):
        return type(obj)(contiguous_features(v) for v in obj)

    return obj
