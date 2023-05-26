# Copyright (c) 2022-2023, InterDigital Communications, Inc
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

""" Main Command-line interface tools for compressai-vision
"""
import argparse
import logging
import sys

from compressai_vision.cli.eval import split_inference

# from compressai_vision.tools import getDataFile, quickLog

COMMANDS = {  # noqa: F405
    "split-inference": split_inference.main,
}

coms = ""
for key in COMMANDS:
    coms += key + ","


def setup_parser():
    common_parser = argparse.ArgumentParser(
        add_help=False, formatter_class=argparse.RawTextHelpFormatter
    )
    common_parser.add_argument(
        "--y", action="store_true", default=False, help="non-interactive run"
    )
    common_parser.add_argument(
        "--debug", action="store_true", default=False, help="debug verbosity"
    )
    common_parser.add_argument(
        "--dump",
        action="store_true",
        default=False,
        help="dump intermediate features",
    )
    common_parser.add_argument(
        "--cpu",
        action="store_true",
        default=False,
        help="never use cuda, just cpu (when applicable)",
    )

    parser = argparse.ArgumentParser(
        description="Includes several subcommands.  For full manual, type compressai-vision manual",
        add_help=True,
    )
    subparsers = parser.add_subparsers(help="select command", dest="command")

    # compression of intermediate data (FC-VCM)
    split_inference.add_subparser(subparsers, parents=[common_parser])

    return parser


def main():
    parser = setup_parser()
    args, unparsed = parser.parse_known_args()
    # print(">",args)
    # return
    for weird in unparsed:
        print("invalid argument", weird)
        raise SystemExit(2)

    if args.command not in COMMANDS.keys():
        print("invalid command", args.command)
        print("subcommands: " + coms)
        sys.exit(2)

    # if args.debug:
    #     loglev = logging.DEBUG
    # else:
    #     loglev = logging.INFO
    # quickLog("split_inference", loglev)

    func = COMMANDS[args.command]
    func(args)  # pass cli args to the function in question


if __name__ == "__main__":
    main()
