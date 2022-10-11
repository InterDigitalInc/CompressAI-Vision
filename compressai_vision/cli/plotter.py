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
import csv
import glob
import json
import os
import sys

import matplotlib.pyplot as plt
import numpy as np

from compressai_vision.tools import getDataFile

colors = ["b", "g", "r", "c", "m", "y", "k", "w"]


def getBaseline(scale):
    path = getDataFile(os.path.join("results", "vtm-scale-" + str(scale) + ".csv"))
    if not os.path.exists(path):
        print("Sorry, can't find file", path)
        sys.exit(2)
    print(path, ":")
    xs = []
    ys = []
    with open(path) as csvfile:
        reader = csv.reader(csvfile, delimiter=" ")
        for cols in reader:
            if "#" in cols[0]:
                print(" ".join(cols[1:]))
                continue  # this is a comment line
            bpp, map_ = float(cols[0]), float(cols[1])
            # print(bpp, map_)
            xs.append(bpp)
            ys.append(map_)
    a = np.array([xs, ys]).transpose()  # (2,6) --> (6,2)
    return a


def jsonFilesToArray(dir_):
    """Reads all json files in a directory.
    The files should have key "qpars" and "bpp".
    Returns numpy array.
    """
    xs = []
    ys = []
    for path in glob.glob(os.path.join(dir_, "*.json")):
        print("reading", path)
        with open(path, "r") as f:
            res = json.load(f)
            # print(res)
            # res has two lists: res["bpp"] & res["map"]: bpp values and corresponding map values
            # assume there is at least res[”bpp"]
            xs += res["bpp"]
            if "map" in res:
                ys += res["map"]
            # print(xs, ys)
    if len(ys) < 1:
        a = np.array(xs).transpose()
        a.sort(0)
        return a
    a = np.array([xs, ys]).transpose()  # (2,6) --> (6,2)
    a.sort(0)
    return a


def tx(ax, st, i, j, color):
    ax.text(
        i,
        j,
        st,
        horizontalalignment="center",
        verticalalignment="center",
        color=color,
        transform=ax.transAxes,
    )


def add_subparser(subparsers, parents=[]):
    subparser = subparsers.add_parser(
        "plot", parents=parents, help="plot mAP-bpp curve"
    )
    required_group = subparser.add_argument_group("required arguments")
    subparser.add_argument(
        "--csv",
        action="store_true",
        default=False,
        help="output result as nicely formated csv table",
    )
    required_group.add_argument(
        "--dirs",
        action="store",
        type=str,
        required=False,
        help="list of directories, each folder contains evaluation result (json files) of certain model done with detectron2-eval",
    )
    """removed:
    subparser.add_argument(
        "--colors",
        action="store",
        type=str,
        required=False,
        help="list of pyplot colors",
    )
    """
    subparser.add_argument(
        "--symbols",
        action="store",
        type=str,
        required=False,
        help="list of pyplot symbols/colors, e.g: o--k,-b, etc.",
    )
    subparser.add_argument(
        "--names", action="store", type=str, required=False, help="list of plot names"
    )
    subparser.add_argument(
        "--eval",
        action="store",
        type=str,
        required=False,
        default=None,
        help="mAP value without (de)compression and pyplot symbol",
    )
    """removed:
    subparser.add_argument(
        "--show-baseline",
        action="store",
        type=int,
        required=False,
        default=None,
        help="show baseline at a certain scale",
    )
    """
    # parsed_args, unparsed_args = parser.parse_known_args()
    # return parsed_args, unparsed_args


def main(p):
    parsed = p
    # for csv and plot needs directory names
    assert parsed.dirs is not None, "needs list of directory names"
    dirs = parsed.dirs.split(",")
    arrays = []
    for dir_ in dirs:
        dir_ = os.path.expanduser(os.path.join(dir_))
        assert os.path.isdir(dir_), "nonexistent dir "+dir_
        arrays.append(jsonFilesToArray(dir_))

    if parsed.csv:
        for dir_, a in zip(dirs, arrays):
            print("\n" + dir_ + ":\n")
            for bpp, map_ in a:
                print(bpp, map_)
        return

    if parsed.command != "plot":
        print("unknow command", parsed.command)
        print("commands are: manual, plot")
        sys.exit(2)

    # assert(parsed.colors is not None), "needs list of pyplot color codes"
    # assert parsed.symbols is not None, "needs list of pyplot symbol codes"
    # assert parsed.names is not None, "needs list of names for plots"
    # let's define some default dummy values instead

    if parsed.symbols is None:
        print("NOTE: you didn't provide a symbol list, will create one instead")
        symbols = []
    else:
        symbols = parsed.symbols.split(",")
    if parsed.names is None:
        print("NOTE: you didn't provide a plot names, will create one instead")
        names = []
    else:
        names = parsed.names.split(",")

    symbols_aux = ["o--k", "-g", "*:r"]
    for i, dir_ in enumerate(dirs):
        if parsed.symbols is None:
            # cyclic:
            symbols.append(symbols_aux[i % len(symbols_aux)])
        if parsed.names is None:
            # names.append("plot"+str(i))
            names.append(dir_.split(os.pathsep)[-1])

    assert (
        len(dirs) == len(symbols) == len(names)
    ), "dirs, symbols and names must have the same length"

    if parsed.eval:
        eval_lis = parsed.eval.split(",")
        if len(eval_lis) < 2:
            print("NOTE: you didn't provide symbol for eval baseline, will make up one")
            eval_lis.append("--c")
        eval_val, eval_symbol = eval_lis
        eval_val = float(eval_val)

    """removed: user has to give this explicitly
    if parsed.show_baseline:
        try:
            names.index("VTM")
        except ValueError:
            pass
        else:
            print("please don't use reserved name VTM")
            sys.exit(2)
        a = getBaseline(parsed.show_baseline)
        arrays.append(a)
        symbols.append("k--*")
        names.append("VTM")
    """

    plt.figure(figsize=(6, 6))

    cc = 0
    for a, symbol, name in zip(arrays, symbols, names):
        # print(a.shape, len(a.shape))
        if len(a.shape) < 2:
            print("mAP value missing, will skip", name)
            continue
        plt.plot(a[:, 0], a[:, 1], symbol)
        ax = plt.gca()
        color_ = None
        for color in colors:  # ["b", "g", ..]
            if color in symbol:  # i.e. if "b" in symbol
                color_ = color
                break
        if not color:
            print("can't resolve color code: please use:", colors)
            sys.exit(2)
        # print(">>", color)
        tx(ax, name, 0.5, 0.50 + cc * 0.05, color_)
        cc += 1

    minx = plt.axis()[0]
    maxx = plt.axis()[1]

    if parsed.eval:
        plt.plot((minx, maxx), (eval_val, eval_val), eval_symbol)
    else:
        print("NOTE: you didn't provide evaluation baseline so will not plot it")

    plt.xlabel("bpp")
    plt.ylabel("mAP")
    print("--> producing out.png to current path")
    plt.savefig(os.path.join("out.png"))
    print("have a nice day!")
    """from the notebook:
    plt.plot(vtm[:,0], vtm[:,1], '*-b', markersize=12)
    plt.plot(coai[:,0], coai[:,1], '.-r')
    plt.plot(mpeg_vcm[:,0], mpeg_vcm[:,1], 'o--k')
    minx=plt.axis()[0]
    maxx=plt.axis()[1]
    plt.plot((minx, maxx), (eval_[:,1], eval_[:,1]), '--g')
    ax = plt.gca()
    tx(ax, "OUR VTM", 0.5, 0.50, "b")
    tx(ax, "COMPRESSAI", 0.5, 0.55, "r")
    tx(ax, "EVAL", 0.5, 0.60, "g")
    tx(ax, "mpeg_vcm VTM", 0.5, 0.65, "k")
    plt.xlabel("bpp")
    plt.ylabel("mAP")
    plt.title("Detection, scale=100%")
    plt.savefig(os.path.join("out.png"))
    """
