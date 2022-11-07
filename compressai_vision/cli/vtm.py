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
import json

from .tools import checkSlice, checkZoo, getQPars, setupVTM

# import os


def add_subparser(subparsers, parents):
    subparser = subparsers.add_parser(
        "vtm", parents=parents, help="generate bitstream with the vtm video encoder"
    )
    required_group = subparser.add_argument_group("required arguments")
    required_group.add_argument(
        "--dataset-name",
        action="store",
        type=str,
        required=False,
        default=None,
        help="name of the dataset",
    )
    subparser.add_argument(
        "--output",
        action="store",
        type=str,
        required=False,
        default="compressai-vision.json",
        help="outputfile, default: compressai-vision.json",
    )

    subparser.add_argument(
        "--vtm_dir",
        action="store",
        type=str,
        required=False,
        default=None,
        help="path to directory with executables EncoderAppStatic & DecoderAppStatic",
    )
    required_group.add_argument(
        "--vtm_cfg",
        action="store",
        type=str,
        required=False,
        default=None,
        help="vtm config file",
    )
    required_group.add_argument(
        "--vtm_cache",
        action="store",
        type=str,
        required=True,
        default=None,
        help="directory to cache vtm bitstreams",
    )
    required_group.add_argument(
        "--qpars",
        action="store",
        type=str,
        required=False,
        default=None,
        help="quality parameters for compressai model or vtm",
    )
    subparser.add_argument(
        "--scale",
        action="store",
        type=int,
        required=False,
        default=100,
        help="image scaling as per VCM working group docs",
    )
    subparser.add_argument(
        "--ffmpeg",
        action="store",
        type=str,
        required=False,
        default="ffmpeg",
        help="ffmpeg command",
    )
    subparser.add_argument(
        "--slice",
        action="store",
        type=str,
        required=False,
        default=None,
        help="use a dataset slice instead of the complete dataset",
    )
    subparser.add_argument(
        "--progressbar",
        action="store_true",
        default=False,
        help="show fancy progressbar",
    )
    subparser.add_argument(
        "--progress",
        action="store",
        type=int,
        required=False,
        default=1,
        help="Print progress this often",
    )
    subparser.add_argument(
        "--tags",
        action="store",
        type=str,
        required=False,
        default=None,
        help="vtm: a list of ids to pick from the dataset/slice",
    )
    subparser.add_argument(
        "--keep",
        action="store_true",
        default=False,
        help="vtm: keep all intermediate files (for debugging)",
    )
    # subparser.add_argument("--dump", action="store_true", default=False) # now in main
    subparser.add_argument(
        "--check",
        action="store_true",
        default=False,
        help="vtm: report if bitstream files are missing",
    )


def main(p):  # noqa: C901
    import cv2

    print("importing fiftyone")
    # fiftyone
    import fiftyone as fo

    ProgressBar = fo.ProgressBar
    print("fiftyone imported")

    # compressai_vision
    from compressai_vision.constant import vf_per_scale

    # from compressai_vision.evaluation.fo import (  # annex predictions from
    #     annexPredictions,
    # )
    from compressai_vision.evaluation.pipeline import VTMEncoderDecoder

    # from compressai_vision.tools import getDataFile

    assert p.dataset_name is not None, "please provide dataset name"
    try:
        dataset = fo.load_dataset(p.dataset_name)
    except ValueError:
        print("FATAL: no such registered database", p.dataset_name)
        return
    assert p.vtm_cache is not None, "need to provide a cache directory"
    assert p.qpars is not None, "need to provide quality parameters for vtm"
    qpars = getQPars(p)

    vtm_encoder_app, vtm_decoder_app, vtm_cfg = setupVTM(p)
    dataset, fr, to = checkSlice(p, dataset)

    # if p.d_list is not None:
    #    print("WARNING: using only certain images from the dataset/slice")

    # use open image ids if avail
    if dataset.get_field("open_images_id"):
        id_field_name = "open_images_id"
    else:
        id_field_name = "id"

    if p.tags is not None:
        lis = p.tags.split(",")  # 0001eeaf4aed83f9,000a1249af2bc5f0
        from fiftyone import ViewField as F

        dataset = dataset.match(F("id_field_name").contains_str(lis))

    if p.scale is not None:
        assert p.scale in vf_per_scale.keys(), "invalid scale value"

    print()
    print("VTM bitstream generation")
    if p.vtm_cache:
        print("WARNING: VTM USES CACHE IN", p.vtm_cache)
    print("Target dir             :", p.vtm_cache)
    print("Quality points/subdirs :", qpars)
    print("Using dataset          :", p.dataset_name)
    print("Image Scaling          :", p.scale)
    if p.slice is not None:
        print("Using slice            :", str(fr) + ":" + str(to))
    if p.tags is not None:
        print("WARNING: Picking samples, based on", id_field_name, "field")
    print("Number of samples      :", len(dataset))
    print("Progressbar            :", p.progressbar)
    print("Output file            :", p.output)
    if p.progressbar and p.progress > 0:
        print("WARNING: progressbar enabled --> disabling normal progress print")
        p.progress = 0
    print("Print progress         :", p.progress)
    if p.keep:
        print("WARNING: keep enabled --> will not remove intermediate files")
    if p.check:
        print(
            "WARNING: checkmode enabled --> will only check if bitstream files exist or not"
            "WARNING: doesn't calculate bbp values either"
        )
    if not p.y:
        input("press enter to continue.. ")

    # save metadata about the run into the json file
    metadata = {
        "dataset": p.dataset_name,
        "just-check": p.check,
        "slice": p.slice,
        "vtm_cache": p.vtm_cache,
        "qpars": qpars,
    }
    with open(p.output, "w") as f:
        f.write(json.dumps(metadata, indent=2))

    xs = []
    for i in qpars:
        print("\nQUALITY PARAMETER", i)
        enc_dec = VTMEncoderDecoder(
            encoderApp=vtm_encoder_app,
            decoderApp=vtm_decoder_app,
            ffmpeg=p.ffmpeg,
            vtm_cfg=vtm_cfg,
            qp=i,
            cache=p.vtm_cache,
            scale=p.scale,
            dump=p.dump,
            skip=p.check,  # if there's a bitstream file then just exit at call to BGR
            keep=p.keep,
            warn=True,
        )
        # with ProgressBar(dataset) as pb: # captures stdout
        if p.progressbar:
            pb = ProgressBar(dataset)

        cc = 0
        """
        if p.checkmode: # just report which bitstreams exist in the cache
            print()
            print("reporting images missing bitstream at '%s'" % enc_dec.getCacheDir())
            print("n / id / open_images_id (use this!) / path")
            check_c=0
        """
        npix_sum = 0
        nbits_sum = 0
        for sample in dataset:
            cc += 1
            # sample.filepath
            path = sample.filepath
            im0 = cv2.imread(path)
            # tag = path.split(os.path.sep)[-1].split(".")[0]  # i.e.: /path/to/some.jpg --> some.jpg --> some
            tag = sample[id_field_name]
            # print(tag)
            nbits, im = enc_dec.BGR(im0, tag=tag)
            if nbits < 0:
                if p.check:
                    print(
                        "WARNING: Bitstream missing for image id={id}, tag={tag}, path={path}".format(
                            id=sample.id, tag=tag, path=path
                        )
                    )
                    continue
                # enc_dec.BGR tried to use the existing bitstream file but failed to decode it
                print(
                    "ERROR: Corrupt data for image id={id}, tag={tag}, path={path}".format(
                        id=sample.id, tag=tag, path=path
                    )
                )
                # .. the bitstream has been removed
                print("ERROR: Trying to regenerate")
                # let's try to generate it again
                nbits, im = enc_dec.BGR(im0, tag=tag)
                if nbits < 0:
                    print(
                        "ERROR: DEFINITELY Corrupt data for image id={id}, tag={tag}, path={path} --> CHECK MANUALLY!".format(
                            id=sample.id, tag=tag, path=path
                        )
                    )
            if not p.check:
                # NOTE: use transformed image im
                npix_sum += im.shape[0] * im.shape[1]
                nbits_sum += nbits
            if p.progress > 0 and ((cc % p.progress) == 0):
                print("sample: ", cc, "/", len(dataset), "tag:", tag)
            if p.progressbar:
                pb.update()

        if not p.check:
            if (nbits_sum < 1) or (npix_sum < 1):
                print("ERROR: nbits_sum", nbits_sum, "npix_sum", npix_sum)
                xs.append(None)
            else:
                xs.append(nbits_sum / npix_sum)

    # print(">>", metadata)
    metadata["bpp"] = xs
    with open(p.output, "w") as f:
        json.dump(metadata, f)

    print("\nDone!\n")
