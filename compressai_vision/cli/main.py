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

# import logging
import sys

from compressai_vision.cli.eval import encdec

# from compressai_vision.tools import getDataFile, quickLog

COMMANDS = {  # noqa: F405
    "encdec": encdec.main,
}

coms = ""
for key in COMMANDS:
    coms += key + ","


def setup_parser():
    parser = argparse.ArgumentParser(
        add_help=False, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        "--y", action="store_true", default=False, help="non-interactive run"
    )
    parser.add_argument(
        "--debug", action="store_true", default=False, help="debug verbosity"
    )
    parser.add_argument(
        "--cpu",
        action="store_true",
        default=False,
        help="never use cuda, just cpu (when applicable)",
    )
    parser.add_argument(
        "--config-path",
        type=str,
        required=True,
        help="A folder of the config files",
    )
    parser.add_argument(
        "--config-name",
        type=str,
        required=True,
        help="Target configuration file",
    )

    return parser


def parse_cmd(argv):
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("command", choices=COMMANDS.keys())
    args = parser.parse_args(argv)
    return args.command


def update_sys_argv(args):
    sys.argv = [
        "",
        f"--config-path={args.config_path}",
        f"--config-name={args.config_name}",
    ]


def main(argv):
    command = parse_cmd(argv[0:1])
    parser = setup_parser()
    args, unparsed = parser.parse_known_args(argv[1:])

    for weird in unparsed:
        print("invalid argument", weird)
        raise SystemExit(2)

    print(f'>> Command << : "{command}"')
    for k, v in vars(args).items():
        print(f">> {k} ".ljust(15), f" = {v}")
    print("\n")

    update_sys_argv(args)

    # [TODO: hyomin.choi & fabien.racapé]
    # if args.debug:
    #     loglev = logging.DEBUG
    # else:
    #     loglev = logging.INFO
    # quickLog("split_inference", loglev)

    # cudnn / benchmark environment setup?

    func = COMMANDS[command]()


if __name__ == "__main__":
    main(sys.argv[1:])
