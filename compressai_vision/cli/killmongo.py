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

"""Kill / clear mongodb 

If there's a runaway mongodb process, etc. or mongodb was terminated unclean.

::

    ~/.fiftyone/var/lib/mongo/mongod.lock # kill mongodb, remove this file

    ~/.fiftyone # kill mongodb & remove the mongodb database .. this can happen if fiftyone & mongo versions are not compatible

    fiftyone.core.service.ServiceListenTimeout: fiftyone.core.service.DatabaseService failed to bind to port

    https://github.com/voxel51/fiftyone/issues/1988     # related github issues
    https://github.com/voxel51/fiftyone/issues/1334
    
"""

import glob
import os
import shutil
import sys
import argparse

def stopMongo():
    print("killing mongo process")
    os.system("killall -9 mongod")
    print("killed mongo process")
    for fname in glob.glob(
        os.path.expanduser(os.path.join("~", ".fiftyone/var/lib/mongo/*lock*"))
    ):
        print("removing", fname, "PRESS ENTER TO CONTINUE")
        input()
        os.remove(fname)


def clearMongo():
    print("killing mongo process")
    os.system("killall -9 mongod")
    print("killed mongo process")
    dirname = os.path.expanduser(os.path.join("~", ".fiftyone"))
    print("WARNING: removing directory", dirname, "PRESS ENTER TO CONTINUE")
    input()
    shutil.rmtree(dirname)


def add_subparser(subparsers, parents=[]):
    subparser = subparsers.add_parser(
        "mongo", parents=parents
    )
    subsubparsers = subparser.add_subparsers(help="select subcommand (stop or clear)", dest="subcommand")
    subsubparsers.add_parser(
        "stop", description="stop local mongodb server and clean lockfiles"
    )
    subsubparsers.add_parser(
        "clear", description="remove the local mongodb database"
    )


def main(p):
    print("killmongo: p=",p)
    if p.subcommand == "stop":
        stopMongo()
    elif p.subcommand == "clear":
        clearMongo()
    """TODO:
    - check if the env variable defining fiftyone mongodb connection is on
    - ..in that case clear using mongoengine
    - report all mongo processes in the system?
    """
