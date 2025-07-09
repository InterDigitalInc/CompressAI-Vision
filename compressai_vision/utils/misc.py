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

import time

from typing import Dict, List, TypeVar

import torch

from torch import Tensor

K = TypeVar("K")
V = TypeVar("V")


def to_cpu(data: Tensor):
    return data.to(torch.device("cpu"))


def time_measure():
    return time.perf_counter()


def dict_sum(a, b):
    c = {}

    for k in set(a) | set(b):
        v = a.get(k, 0) + b.get(k, 0)
        c[k] = v

    return c


def ld_to_dl(ld: List[Dict[K, V]]) -> Dict[K, List[V]]:
    """Converts a list of dicts into a dict of lists."""
    dl = {}
    for d in ld:
        for k, v in d.items():
            if k not in dl:
                dl[k] = []
            dl[k].append(v)
    return dl


def dl_to_ld(dl: Dict[K, List[V]]) -> List[Dict[K, V]]:
    """Converts a dict of lists into a list of dicts."""
    if not dl:
        return []
    expected_len = len(next(iter(dl.values())))
    assert all(len(v) == expected_len for v in dl.values())
    ld = [dict(zip(dl.keys(), v)) for v in zip(*dl.values())]
    return ld


class metric_tracking:
    def __init__(self):
        self._buffer = []

    def append(self, d):
        self._buffer.append(d)

    @property
    def avg(self):
        return sum(self._buffer) / len(self._buffer)

    @property
    def sum(self):
        return sum(self._buffer)
