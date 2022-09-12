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

import inspect
import logging
import os
import subprocess


def confLogger(logger, level):
    logger.setLevel(level)
    # print("confLogger", logger.handlers)
    # print("confLogger", logger.hasHandlers())
    # if not logger.hasHandlers(): # when did this turn into a practical joke?
    if len(logger.handlers) < 1:
        formatter = logging.Formatter("%(name)s - %(levelname)s - %(message)s")
        ch = logging.StreamHandler()
        ch.setFormatter(formatter)
        logger.addHandler(ch)


def quickLog(name, level):
    logger = logging.getLogger(name)
    confLogger(logger, level)
    return logger


def pathExists(p):
    return os.path.exists(os.path.expanduser(p))


def getModulePath():
    lis = inspect.getabsfile(inspect.currentframe()).split("/")
    st = "/"
    for f in lis[:-1]:
        st = os.path.join(st, f)
    return st


def getDataPath():
    return os.path.join(getModulePath(), "data")


def getDataFile(fname):
    """Return complete path to datafile fname.  Data files are in the directory compressai_vision/"""
    return os.path.join(getDataPath(), fname)


def test_command(comm):
    try:
        subprocess.Popen(
            comm, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )
    except FileNotFoundError:
        raise
    else:
        return comm
