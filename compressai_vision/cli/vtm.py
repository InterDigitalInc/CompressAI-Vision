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

"""cli detectron2_eval functionality
"""
import copy, os, uuid, datetime
import json
import os
import cv2

# fiftyone
import fiftyone as fo
from fiftyone import ProgressBar

# compressai_vision
from compressai_vision.evaluation.fo import annexPredictions  # annex predictions from
from compressai_vision.evaluation.pipeline import (
    CompressAIEncoderDecoder,
    VTMEncoderDecoder,
)
from compressai_vision.tools import getDataFile


def main(p):
    assert p.name is not None, "please provide dataset name"
    try:
        dataset = fo.load_dataset(p.name)
    except ValueError:
        print("FATAL: no such registered database", p.name)
        return
    assert p.vtm_cache is not None, "need to provide a cache directory"
    assert p.qpars is not None, "need to provide quality parameters for vtm"
    try:
        qpars = [float(i) for i in p.qpars.split(",")]
    except Exception as e:
        print("problems with your quality parameter list")
        raise e
    if p.vtm_dir is None:
        try:
            vtm_dir = os.environ["VTM_DIR"]
        except KeyError as e:
            print("please define --vtm_dir or set environmental variable VTM_DIR")
            # raise e
            return
    else:
        vtm_dir = p.vtm_dir

    vtm_dir = os.path.expanduser(vtm_dir)

    if p.vtm_cfg is None:
        vtm_cfg = getDataFile("encoder_intra_vtm_1.cfg")
        print("WARNING: using VTM default config file", vtm_cfg)
    else:
        vtm_cfg = p.vtm_cfg

    vtm_cfg = os.path.expanduser(vtm_cfg)  # some more systematic way of doing these..

    assert os.path.isfile(vtm_cfg), "vtm config file not found"

    # try both filenames..
    vtm_encoder_app = os.path.join(vtm_dir, "EncoderAppStatic")
    if not os.path.isfile(vtm_encoder_app):
        vtm_encoder_app = os.path.join(vtm_dir, "EncoderAppStaticd")
    if not os.path.isfile(vtm_encoder_app):
        print("FATAL: can't find EncoderAppStatic(d) in", vtm_dir)
    # try both filenames..
    vtm_decoder_app = os.path.join(vtm_dir, "DecoderAppStatic")
    if not os.path.isfile(vtm_decoder_app):
        vtm_decoder_app = os.path.join(vtm_dir, "DecoderAppStaticd")
    if not os.path.isfile(vtm_decoder_app):
        print("FATAL: can't find DecoderAppStatic(d) in", vtm_dir)

    print()
    print("VTM bitstream generation")
    print("Target dir             :", p.vtm_cache)
    print("Quality points/subdirs :", qpars)
    print("Using dataset          :", p.name)
    print("Number of samples      :", len(dataset))
    if not p.y:
        input("press enter to continue.. ")
    for i in qpars:
        print("\nQUALITY PARAMETER", i)
        enc_dec = VTMEncoderDecoder(
            encoderApp=vtm_encoder_app,
            decoderApp=vtm_decoder_app,
            ffmpeg=p.ffmpeg,
            vtm_cfg=vtm_cfg,
            qp=i,
            cache=p.vtm_cache,
        )
        with ProgressBar(dataset) as pb:
            for sample in dataset:
                # sample.filepath
                path = sample.filepath
                im = cv2.imread(path)
                tag = path.split(os.path.sep)[-1].split(".")[
                    0
                ]  # i.e.: /path/to/some.jpg --> some.jpg --> some
                # print(tag)
                bpp, im = enc_dec.BGR(im, tag=tag)
                pb.update()
    print("\nHAVE A NICE DAY!\n")
