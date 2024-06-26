# Copyright (c) 2022-2024 InterDigital Communications, Inc
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

"""cli metrics-eval functionality
"""
# import copy
# import datetime
import json
import math

from .tools import checkSlice, getQPars, loadEncoderDecoderFromPath, setupVTM


def add_subparser(subparsers, parents):
    subparser = subparsers.add_parser(
        "metrics-eval",
        parents=parents,
        help="evaluate model with psnr and ms-ssim",
    )
    required_group = subparser.add_argument_group("required arguments")
    compressai_group = subparser.add_argument_group("compressai-zoo arguments")
    vtm_group = subparser.add_argument_group("vtm arguments")
    optional_group = subparser.add_argument_group("optional arguments")
    required_group.add_argument(
        "--dataset-name",
        action="store",
        type=str,
        required=True,
        default=None,
        help="name of the dataset",
    )
    optional_group.add_argument(
        "--output",
        action="store",
        type=str,
        required=False,
        default="compressai-vision.json",
        help="outputfile. Default: compressai-vision.json",
    )
    compressai_group.add_argument(
        "--compressai-model-name",
        action="store",
        type=str,
        required=False,
        default=None,
        help="name of an existing model in compressai-zoo. Example: 'cheng2020-attn' ",
    )
    compressai_group.add_argument(
        "--compression-model-path",
        action="store",
        type=str,
        required=False,
        default=None,
        help="a path to a directory containing model.py for custom development model",
    )
    vtm_group.add_argument(
        "--vtm",
        action="store_true",
        default=False,
        help="To enable vtm codec. default: False",
    )
    vtm_group.add_argument(
        "--vtm_dir",
        action="store",
        type=str,
        required=False,
        default=None,
        help="path to directory with executables EncoderAppStatic & DecoderAppStatic",
    )
    vtm_group.add_argument(
        "--vtm_cfg",
        action="store",
        type=str,
        required=False,
        default=None,
        help="vtm config file. Example: 'encoder_intra_vtm.cfg' ",
    )
    vtm_group.add_argument(
        "--vtm_cache",
        action="store",
        type=str,
        required=False,
        default=None,
        help="directory to cache vtm bitstreams",
    )
    optional_group.add_argument(
        "--qpars",
        action="store",
        type=str,
        required=False,
        default=None,
        help="quality parameters for compressai model or vtm. For compressai-zoo model, it should be integer 1-8. For VTM, it should be integer from 0-51.",
    )
    optional_group.add_argument(
        "--scale",
        action="store",
        type=int,
        required=False,
        default=100,
        help="image scaling as per VCM working group docs. Default: 100",
    )
    optional_group.add_argument(
        "--ffmpeg",
        action="store",
        type=str,
        required=False,
        default="ffmpeg",
        help="path of ffmpeg executable. Default: ffmpeg",
    )
    optional_group.add_argument(
        "--slice",
        action="store",
        type=str,
        required=False,
        default=None,
        help="use a dataset slice instead of the complete dataset. Example: 0:2 for the first two images",
    )
    # subparser.add_argument("--debug", action="store_true", default=False) # not here
    optional_group.add_argument(
        "--progressbar",
        action="store_true",
        default=False,
        help="show fancy progressbar. Default: False",
    )
    optional_group.add_argument(
        "--progress",
        action="store",
        type=int,
        required=False,
        default=1,
        help="Print progress this often",
    )
    return subparser


def main(p):  # noqa: C901
    # check that only one is defined
    defined_codec = ""
    for codec in [p.compressai_model_name, p.vtm, p.compression_model_path]:
        if codec:
            if defined_codec:  # second match!
                raise AssertionError(
                    "please define only one of the following: compressai_model_name, vtm or compression_model_path"
                )
            defined_codec = codec

    assert p.dataset_name is not None, "please provide dataset name"
    # fiftyone
    print("importing fiftyone")
    import fiftyone as fo

    # dataset.clone needs this
    # from compressai_vision import patch  # noqa: F401
    print("fiftyone imported")
    from compressai_vision.evaluation.pipeline import (
        CompressAIEncoderDecoder,
        VTMEncoderDecoder,
    )
    from compressai_vision.pipelines.fo_vcm.constant import vf_per_scale

    try:
        dataset = fo.load_dataset(p.dataset_name)
    except ValueError:
        print("FATAL: no such registered dataset", p.dataset_name)
        return

    dataset, fr, to = checkSlice(p, dataset)

    # print(">", p.compressai_model_name, p.vtm, p.compression_model_path, p.qpars)
    if (
        (p.compressai_model_name is None)
        and (p.vtm is False)
        and (p.compression_model_path is None)
    ):
        print("please provide compressai_model_name, compression_model_path or use vtm")
        return

    # check quality parameter list
    assert p.qpars is not None, "need to provide integer quality parameters"
    qpars = getQPars(p)

    if p.compressai_model_name is not None:  # compression from compressai zoo
        from compressai import zoo

        compression_model = getattr(zoo, p.compressai_model_name)

    elif p.compression_model_path is not None:
        encoder_decoder_func = loadEncoderDecoderFromPath(p.compression_model_path)

    elif p.vtm:  # setup VTM
        vtm_encoder_app, vtm_decoder_app, vtm_cfg = setupVTM(p)

    # *** CHOOSE COMPRESSION SCHEME OK ***

    if p.scale is not None:
        assert p.scale in vf_per_scale.keys(), "invalid scale value"

    """
    try:
        username = os.environ["USER"]
    except KeyError:
        username = "nouser"
    tmp_name0 = p.dataset_name + "-{0:%Y-%m-%d-%H-%M-%S-%f}".format(
        datetime.datetime.now()
    )

    tmp_name = "detectron-run-{username}-{tmp_name0}".format(
        username=username, tmp_name0=tmp_name0
    )
    """
    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if p.no_cuda:
        device = "cpu"
    print()
    print("Using dataset          :", p.dataset_name)
    # print("Dataset tmp clone      :", tmp_name)
    print("Image scaling          :", p.scale)
    if p.slice is not None:  # can't use slicing
        print("WARNING: Using slice   :", str(fr) + ":" + str(to))
    print("Number of samples      :", len(dataset))
    print("Torch device           :", device)
    if p.compressai_model_name is not None:
        print("Using compressai model :", p.compressai_model_name)
    elif p.compression_model_path is not None:
        print("Using custom model.py from")
        print("                       :", p.compression_model_path)
    elif p.vtm:
        print("Using VTM               ")
        if p.vtm_cache:
            # assert(os.path.isdir(p.vtm_cache)), "no such directory "+p.vtm_cache
            # ..created by the VTMEncoderDecoder class
            print("WARNING: VTM USES CACHE IN", p.vtm_cache)
    print("Quality parameters     :", qpars)
    print("Progressbar            :", p.progressbar)
    if p.progressbar and p.progress > 0:
        print("WARNING: progressbar enabled --> disabling normal progress print")
        p.progress = 0
    print("Print progress         :", p.progress)
    print("Output file            :", p.output)
    if p.dump:
        print("WARNING - dump enabled : will dump intermediate images")

    if not p.y:
        input("press enter to continue.. ")

    # save metadata about the run into the json file
    metadata = {
        "dataset": p.dataset_name,
        # "tmp datasetname": tmp_name,
        "slice": p.slice,
        "codec": defined_codec,
        "qpars": qpars,
    }
    with open(p.output, "w") as f:
        f.write(json.dumps(metadata, indent=2))

    """
    # please see ../monkey.py for problems I encountered when cloning datasets
    # simultaneously with various multiprocesses/batch jobs
    print("cloning dataset", p.dataset_name, "to", tmp_name)
    dataset = dataset.clone(tmp_name)
    dataset.persistent = True
    # fo.core.odm.database.sync_database() # this would've helped? not sure..
    """
    psnr_lis = []
    mssim_lis = []
    bpp_lis = []

    import traceback

    import cv2

    # use open image ids if avail
    if dataset.get_field("open_images_id"):
        id_field_name = "open_images_id"
    else:
        id_field_name = "id"

    for quality in qpars:
        if (
            p.compressai_model_name or p.compression_model_path
        ):  # compressai model, either from the zoo or from a directory:
            if p.compressai_model_name is not None:
                # e.g. "bmshj2018-factorized"
                print("\nQUALITY PARAMETER: ", quality)
                net = (
                    compression_model(quality=quality, pretrained=True)
                    .eval()
                    .to(device)
                )
                enc_dec = CompressAIEncoderDecoder(
                    net, device=device, scale=p.scale, ffmpeg=p.ffmpeg, dump=p.dump
                )
            else:  # or a custom model from a file:
                enc_dec = encoder_decoder_func(
                    quality=quality,
                    device=device,
                    scale=p.scale,
                    ffmpeg=p.ffmpeg,
                    dump=p.dump,
                )
                # net = compression_model(quality=quality).eval().to(device)
                # make sure we load just trained models and pre-trained/ updated entropy parameters

        elif p.vtm:
            raise (BaseException("metrics calc for VTM not yet implemented"))
            enc_dec = VTMEncoderDecoder(
                encoderApp=vtm_encoder_app,
                decoderApp=vtm_decoder_app,
                ffmpeg=p.ffmpeg,
                vtm_cfg=vtm_cfg,
                qp=quality,
                cache=p.vtm_cache,
                scale=p.scale,
                warn=True,
            )
        else:
            raise BaseException("program logic error")

        npix_sum = 0
        nbits_sum = 0
        psnr_sum = 0
        mssim_sum = 0

        from fiftyone import ProgressBar

        if p.progressbar:
            pb = ProgressBar(dataset)

        cc = 0
        for sample in dataset:
            path = sample.filepath
            im = cv2.imread(path)
            tag = sample[id_field_name]
            if im is None:
                print("FATAL: could not read the image file '" + path + "'")
                return -1
            try:
                nbits, im_ = enc_dec.BGR(
                    im, tag=tag
                )  # include a tag for cases where EncoderDecoder uses caching
            except Exception as e:
                print("EncoderDecoder failed with '" + str(e) + "'")
                print("Traceback:")
                traceback.print_exc()
                return -1
            if nbits < 0:
                # there's something wrong with the encoder/decoder process
                # say, corrupt data from the VTMEncode bitstream etc.
                print("EncoderDecoder returned error: will try using it once again")
                nbits, im_ = enc_dec.BGR(im, tag=tag)
            if nbits < 0:
                print("EncoderDecoder returned error - again!  Will abort calculation")
                return -1

            npix_sum += im_.shape[0] * im_.shape[1]
            nbits_sum += nbits
            psnr, mssim = enc_dec.getMetrics()
            # print(">cc", cc)
            # print(">tag", tag)
            # print(">psnr, mssim", psnr, mssim, type(psnr))
            if math.isnan(psnr) or math.isnan(mssim):
                print(
                    "getMetrics returned nan - you images are probably corrupt.  Will exit now."
                )
                return

            psnr_sum += psnr
            mssim_sum += mssim
            if p.progressbar:
                pb.update()
            elif p.progress > 0 and ((cc % p.progress) == 0):
                print("sample: ", cc, "/", len(dataset) - 1)
            cc += 1

        bpp = nbits_sum / npix_sum
        psnr = psnr_sum / len(dataset)
        mssim = mssim_sum / len(dataset)
        bpp_lis.append(bpp)
        psnr_lis.append(psnr)
        mssim_lis.append(mssim)

    # print(">>", metadata)
    metadata["bpp"] = bpp_lis
    metadata["psnr"] = psnr_lis
    metadata["mssim"] = mssim_lis
    with open(p.output, "w") as f:
        f.write(json.dumps(metadata, indent=2))

    """
    # remove the tmp database
    print("deleting tmp database", tmp_name)
    fo.delete_dataset(tmp_name)
    """
    print("\nDone!\n")
    """load with:
    with open(p.output,"r") as f:
        res=json.load(f)
    """
