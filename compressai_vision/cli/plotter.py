 
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
import os, json, glob
import numpy as np
import matplotlib.pyplot as plt

from compressai_vision.tools import quickLog, getDataFile


def jsonFilesToArray(dir_):
    """Reads all json files in a directory.
    The files should have key "qpars" and "bpp".
    Returns numpy array.
    """
    xs=[]; ys=[]
    for path in glob.glob(os.path.join(dir_, "*.json")):
        print("reading", path)
        with open(path, "r") as f:
            res=json.load(f)
            # print(res)
            # res has two lists: res["bpp"] & res["map"]: bpp values and corresponding map values
            # assume there is at least res[”bpp"]
            xs += res["bpp"]
            if "map" in res:
                ys += res["map"]
            # print(xs, ys)
    if len(ys) < 1:
        a=np.array(xs).transpose()
        a.sort(0)
        return a
    a=np.array([xs, ys]).transpose() # (2,6) --> (6,2)
    a.sort(0)
    return a


def process_cl_args():
    parser = argparse.ArgumentParser(
        usage=(
            "compressai-vision-plot [options] command\n"
            "\n"
            "please use the command 'manual' for full documentation of this program\n"
            "\n"
        )
    )
    parser.add_argument("command", action="store", type=str, help="mandatory command")

    parser.add_argument(
        "--dirs",
        action="store",
        type=str,
        required=True,
        help="list of directories",
    )
    parser.add_argument(
        "--colors",
        action="store",
        type=str,
        required=True,
        help="list of pyplot colors",
    )
    parser.add_argument(
        "--symbols",
        action="store",
        type=str,
        required=True,
        help="list of pyplot symbols",
    )
    parser.add_argument(
        "--names",
        action="store",
        type=str,
        required=True,
        help="list of plot names"
    )
    parser.add_argument(
        "--show-baseline",
        action="store",
        type=int,
        required=False,
        default=None,
        help="show baseline at a certain scale"
    )
    
    parsed_args, unparsed_args = parser.parse_known_args()
    return parsed_args, unparsed_args

def main():
    parsed, unparsed = process_cl_args()

    for weird in unparsed:
        print("invalid argument", weird)
        raise SystemExit(2)

    if parsed.command == "manual":
        with open(getDataFile("plotter.txt"), "r") as f:
            print(f.read())
        return

    dirs=parsed.dirs.split(",")
    colors=parsed.colors.split(",")
    symbols=parsed.symbols.split(",")
    names=parsed.names.split(",")
    show_baseline=parsed.show_baseline

    assert len(dirs)==len(colors)==len(symbols)==len(names),\
        "dirs, colors, symbols and names must have the same length"

    arrays=[]
    for dir_ in dirs:
        dir_ = os.path.expanduser("~")
        assert os.path.isdir(dir_), "nonexistent dir"
        arrays.append(
            jsonFilesToArray(dir_)
        )

    plt.figure(figsize=(6, 6))
    """
    plt.plot(vtm[:,0], vtm[:,1], '*-b', markersize=12)
    plt.plot(coai[:,0], coai[:,1], '.-r')
    plt.plot(nokia[:,0], nokia[:,1], 'o--k')
    minx=plt.axis()[0]
    maxx=plt.axis()[1]
    plt.plot((minx, maxx), (eval_[:,1], eval_[:,1]), '--g')
    ax = plt.gca()
    tx(ax, "OUR VTM", 0.5, 0.50, "b")
    tx(ax, "COMPRESSAI", 0.5, 0.55, "r")
    tx(ax, "EVAL", 0.5, 0.60, "g")
    tx(ax, "NOKIA VTM", 0.5, 0.65, "k")
    plt.xlabel("bpp")
    plt.ylabel("mAP")
    plt.title("Detection, scale=100%")
    plt.savefig(os.path.join("out.png"))
    """

