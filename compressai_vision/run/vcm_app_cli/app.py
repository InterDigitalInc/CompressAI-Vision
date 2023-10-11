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

"""Launch fiftyone app
"""
# import os
import time


def add_subparser(subparsers, parents):
    subparser = subparsers.add_parser(
        "app",
        parents=parents,
        help="launch the celebrated fiftyone app for dataset visualization",
    )
    req_group = subparser.add_argument_group("required arguments")
    req_group.add_argument(
        "--dataset-name",
        action="store",
        type=str,
        required=True,
        default=None,
        help="name of the dataset",
    )
    opt_group = subparser.add_argument_group("option arguments")
    opt_group.add_argument(
        "--address",
        action="store",
        type=str,
        required=False,
        default=None,
        help="address to bind the webapp to",
    )
    opt_group.add_argument(
        "--port",
        action="store",
        type=int,
        required=False,
        default=None,
        help="port to bind the webapp to",
    )


def main(p):
    print("importing fiftyone")
    import fiftyone as fo

    print("fiftyone imported")
    print()

    if p.dataset_name not in fo.list_datasets():
        print("No such dataset", p.dataset_name)
        return 2
    dataset = fo.load_dataset(p.dataset_name)

    if p.address is None:
        p.address = fo.config.default_app_address
    if p.port is None:
        p.port = fo.config.default_app_port
    print("Launching app at address %s port %i" % (p.address, p.port))
    print("press CTRL-C to terminate")
    print()
    # print("Here is your link:")
    # print()
    # print("https://%s:%i" % (p.address, p.port))
    # print()
    fo.launch_app(dataset=dataset, address=p.address, port=p.port)
    while True:
        try:
            time.sleep(1)
        except KeyboardInterrupt:
            break
    fo.close_app()
    print("Have a nice day!")
