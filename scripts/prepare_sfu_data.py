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

import argparse
import json
from pathlib import Path
import os
import sys


def setup_args():
    parser = argparse.ArgumentParser(description="")
    parser.add_argument(
        "-o", "--outputdir", default="", type=str, help="Output directory"
    )
    parser.add_argument(
        "--ffmpeg",
        default="ffmpeg",
        type=str,
        help=(
            "ffmpeg command, can include grid command prefix and docker image e.g.,"
            " sbatch --cpus 1 --image sellorm/ffmpeg:4.2.2 -- ffmpeg (default: %(default)s)"
        ),
    )
    return parser


def main(argv):
    args = setup_args().parse_args(argv)

    # get sequence info
    sfu_sequences_info = Path(
        f"{Path(__file__).resolve().parent}/../data/mpeg-fcvcm/sfu-configs.json"
    )
    with sfu_sequences_info.open("r") as f:
        try:
            data = json.load(f)
        except json.decoder.JSONDecodeError as err:
            print(f'Error reading file "{sfu_sequences_info}"')
            raise err

    # output directory for PNGs and labels
    if args.outputdir == "":
        sfu_png_dir = Path(
            f"{Path(__file__).resolve().parent}/../data/mpeg-fcvcm/SFU_pngs"
        )
    else:
        sfu_png_dir = Path(args.outputdir)
    sfu_png_dir.mkdir(parents=True, exist_ok=True)

    for class_name, class_data in data.items():
        for seq_name, seq_data in class_data["seqs"].items():
            width, height = class_data["resolution"]
            nb_frames = seq_data["LastFrame"] - seq_data["FirstFrame"] + 1

            seq_dir = sfu_png_dir / seq_name
            seq_dir.mkdir(parents=True, exist_ok=True)
            # COCO structure:
            images_dir = seq_dir / "images"
            annotations_dir = seq_dir / "annotations"
            annotations_dir.mkdir(parents=True, exist_ok=True)
            images_dir.mkdir(parents=True, exist_ok=True)
            print(args.ffmpeg)
            cmd = f'{args.ffmpeg} -s {width}x{height} -r {seq_data["FrameRate"]} -pix_fmt yuv420p -i {seq_data["path"]} -vf select="gte(n\, {seq_data["FirstFrame"]})" -start_number 0 -vframes {nb_frames} {images_dir.resolve()}/{seq_name}_%3d.png'
            print(cmd)
            os.system(cmd)


if __name__ == "__main__":
    main(sys.argv[1:])
