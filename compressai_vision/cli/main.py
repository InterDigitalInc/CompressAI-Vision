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
import argparse
import logging

from compressai_vision.tools import quickLog, getDataFile

# import configparser  # https://docs.python.org/3/library/configparser.html


def process_cl_args():
    # def str2bool(v):
    #     return v.lower() in ("yes", "true", "t", "1")

    # about black: https://github.com/psf/black/issues/397
    parser = argparse.ArgumentParser(
        usage=(
            "compressai-vision [options] command\n"
            "\n"
            "please use the command 'manual' for full documentation of this program\n"
            "\n"
        )
    )
    # parser.register('type','bool',str2bool)  # this works only in theory..
    parser.add_argument("command", action="store", type=str, help="mandatory command")

    parser.add_argument(
        "--name",
        action="store",
        type=str,
        required=False,
        default=None,
        help="name of the dataset",
    )
    parser.add_argument(
        "--lists",
        action="store",
        type=str,
        required=False,
        default=None,
        help="comma-separated list of list files",
    )
    parser.add_argument(
        "--split",
        action="store",
        type=str,
        required=False,
        default="validation",
        help="database sub-name, say, 'train' or 'validation'",
    )

    parser.add_argument(
        "--dir",
        action="store",
        type=str,
        required=False,
        default=None,
        help="target/source directory, depends on command",
    )
    parser.add_argument(
        "--target_dir",
        action="store",
        type=str,
        required=False,
        default=None,
        help="target directory for nokia_convert",
    )
    parser.add_argument(
        "--label",
        action="store",
        type=str,
        required=False,
        default=None,
        help="nokia-formatted image-level labels",
    )
    parser.add_argument(
        "--bbox",
        action="store",
        type=str,
        required=False,
        default=None,
        help="nokia-formatted bbox data",
    )
    parser.add_argument(
        "--mask",
        action="store",
        type=str,
        required=False,
        default=None,
        help="nokia-formatted segmask data",
    )

    parser.add_argument(
        "--type",
        action="store",
        type=str,
        required=False,
        default="OpenImagesV6Dataset",
        help="image set type to be imported",
    )

    parser.add_argument(
        "--proto",
        action="store",
        type=str,
        required=False,
        default=None,
        help="evaluation protocol",
    )
    parser.add_argument(
        "--model",
        action="store",
        type=str,
        required=False,
        default=None,
        help="detectron2 model",
    )
    parser.add_argument(
        "--output",
        action="store",
        type=str,
        required=False,
        default="compressai-vision.json",
        help="results output file",
    )

    parser.add_argument(
        "--compressai",
        action="store",
        type=str,
        required=False,
        default=None,
        help="use compressai model",
    )
    parser.add_argument("--vtm", action="store_true", default=False)
    parser.add_argument(
        "--qpars",
        action="store",
        type=str,
        required=False,
        default=None,
        help="quality parameters for compressai model or vtm",
    )
    parser.add_argument(
        "--scale",
        action="store",
        type=int,
        required=False,
        default=100,
        help="image scaling as per VCM working group docs",
    )
    parser.add_argument(
        "--vtm_dir",
        action="store",
        type=str,
        required=False,
        default=None,
        help="path to directory with executables EncoderAppStatic & DecoderAppStatic",
    )
    parser.add_argument(
        "--ffmpeg",
        action="store",
        type=str,
        required=False,
        default="ffmpeg",
        help="ffmpeg command",
    )
    parser.add_argument(
        "--vtm_cfg",
        action="store",
        type=str,
        required=False,
        default=None,
        help="vtm config file",
    )
    parser.add_argument(
        "--vtm_cache",
        action="store",
        type=str,
        required=False,
        default=None,
        help="directory to cache vtm bitstreams",
    )
    parser.add_argument(
        "--slice",
        action="store",
        type=str,
        required=False,
        default=None,
        help="use a dataset slice instead of the complete dataset",
    )
    parser.add_argument(
        "--progress",
        action="store",
        type=int,
        required=False,
        default=1,
        help="Print progress this often",
    )
    parser.add_argument(
        "--debug", action="store_true", default=False, help="debug verbosity"
    )
    parser.add_argument(
        "--keep", action="store_true", default=False, help="vtm: keep all intermediate files (for debugging)"
    )
    parser.add_argument(
        "--checkmode", action="store_true", default=False, help="vtm: report if bitstream files are missing"
    )
    parser.add_argument(
        "--tags",
        action="store",
        type=str,
        required=False,
        default=None,
        help="vtm: a list of open_image_ids to pick from the dataset/slice"
    )
    parser.add_argument(
        "--dump",
        action="store_true",
        default=False,
        help="debugging: dump intermediate data to local directory",
    )
    parser.add_argument(
        "--y", action="store_true", default=False, help="non-interactive run"
    )
    parser.add_argument(
        "--progressbar",
        action="store_true",
        default=False,
        help="show fancy progressbar",
    )
    parser.add_argument("--mock", action="store_true", default=False, help="mock tests")
    parsed_args, unparsed_args = parser.parse_known_args()
    return parsed_args, unparsed_args


def main():
    parsed, unparsed = process_cl_args()

    for weird in unparsed:
        print("invalid argument", weird)
        raise SystemExit(2)

    # setting loglevels manually
    # should only be done in tests:
    """
    logger = logging.getLogger("name.space")
    confLogger(logger, logging.INFO)
    """
    if parsed.debug:
        loglev = logging.DEBUG
    else:
        loglev = logging.INFO
    quickLog("CompressAIEncoderDecoder", loglev)
    quickLog("VTMEncoderDecoder", loglev)

    if parsed.command == "manual":
        with open(getDataFile("manual.txt"), "r") as f:
            print(f.read())
        return

    # parameter filtering/mods
    if parsed.scale == 0:
        parsed.scale = None

    # some command filtering here
    if parsed.command in [
        "download",
        "list",
        "dummy",
        "deregister",
        "nokia_convert",
        "register",
        "detectron2_eval",
        "load_eval",
        "vtm",
    ]:
        from compressai_vision import cli

        # print("command is", parsed.command)
        func = getattr(cli, parsed.command)
        func(parsed)
    else:
        print("unknown command", parsed.command)
        raise SystemExit(2)
    # some ideas on how to handle config files & default values
    #
    # this directory is ~/.skeleton/some_data/ :
    # init default data with yaml constant string
    """
    some_data_dir = AppLocalDir("some_data")
    if (not some_data_dir.has("some.yml")) or parsed.reset:
        with open(some_data_dir.getFile("some.yml"), "w") as f:
            f.write(constant.SOME)
    """


if __name__ == "__main__":
    main()
