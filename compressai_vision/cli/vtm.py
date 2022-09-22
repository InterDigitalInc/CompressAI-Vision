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
import os


def main(p):
    import cv2

    # fiftyone
    import fiftyone as fo
    from fiftyone import ProgressBar

    # compressai_vision
    from compressai_vision.evaluation.fo import (
        annexPredictions,
    )  # annex predictions from
    from compressai_vision.evaluation.pipeline import (
        CompressAIEncoderDecoder,
        VTMEncoderDecoder,
    )
    from compressai_vision.tools import getDataFile
    from compressai_vision.constant import vf_per_scale

    assert p.name is not None, "please provide dataset name"
    try:
        dataset = fo.load_dataset(p.name)
    except ValueError:
        print("FATAL: no such registered database", p.name)
        return
    assert p.vtm_cache is not None, "need to provide a cache directory"
    assert p.qpars is not None, "need to provide quality parameters for vtm"
    try:
        # qpars = [float(i) for i in p.qpars.split(",")]
        qpars = [int(i) for i in p.qpars.split(",")]  # integer for god's sake!
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

    if p.slice is not None:
        print("WARNING: using a dataset slice instead of full dataset")
        # say, 0:100
        nums = p.slice.split(":")
        if len(nums) < 2:
            print("invalid slicing: use normal python slicing, say, 0:100")
            return
        try:
            fr = int(nums[0])
            to = int(nums[1])
        except ValueError:
            print("invalid slicing: use normal python slicing, say, 0:100")
            return
        assert to > fr, "invalid slicing: use normal python slicing, say, 0:100"
        dataset = dataset[fr:to]

    # if p.d_list is not None:
    #    print("WARNING: using only certain images from the dataset/slice")

    if p.tags is not None:
        lis = p.tags.split(",")  # 0001eeaf4aed83f9,000a1249af2bc5f0
        from fiftyone import ViewField as F
        dataset = dataset.match(F("open_images_id").contains_str(lis))

    if p.scale is not None:
        assert p.scale in vf_per_scale.keys(), "invalid scale value"

    print()
    print("VTM bitstream generation")
    if p.vtm_cache:
        print("WARNING: VTM USES CACHE IN", p.vtm_cache)
    print("Target dir             :", p.vtm_cache)
    print("Quality points/subdirs :", qpars)
    print("Using dataset          :", p.name)
    print("Image Scaling          :", p.scale)
    if p.slice is not None:
        print("Using slice            :", str(fr) + ":" + str(to))
    if p.tags is not None:
        print("WARNING: Picking samples, based on open_images_id field")
    print("Number of samples      :", len(dataset))
    print("Progressbar            :", p.progressbar)
    if p.progressbar and p.progress > 0:
        print("WARNING: progressbar enabled --> disabling normal progress print")
        p.progress = 0
    print("Print progress         :", p.progress)
    if p.keep:
        print("WARNING: keep enabled --> will not remove intermediate files")
    if p.check:
        print(
            "WARNING: checkmode enabled --> will only check if bitstream files exist or not"
        )
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
            scale=p.scale,
            dump=p.dump,
            skip=p.check,  # if there's a bitstream file then just exit at call to BGR
            keep=p.keep,
            warn=True
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
        for sample in dataset:
            cc += 1
            # sample.filepath
            path = sample.filepath
            im0 = cv2.imread(path)
            # tag = path.split(os.path.sep)[-1].split(".")[0]  # i.e.: /path/to/some.jpg --> some.jpg --> some
            tag = (
                sample.open_images_id
            )  # TODO: if there is no open_images_id, then use the normal id?
            # print(tag)
            nbits, im = enc_dec.BGR(im0, tag=tag)
            if nbits < 0:
                if p.check:
                    print(
                        "Bitstream missing for image id={id}, openImageId={tag}, path={path}".format(
                            id=sample.id, tag=tag, path=path
                        )
                    )
                    continue
                # enc_dec.BGR tried to use the existing bitstream file but failed to decode it
                print(
                    "Corrupt data for image id={id}, openImageId={tag}, path={path}".format(
                        id=sample.id, tag=tag, path=path
                    )
                )
                # .. the bitstream has been removed
                print("Trying to regenerate")
                # let's try to generate it again
                nbits, im = enc_dec.BGR(im0, tag=tag)
                if nbits < 0:
                    print(
                        "DEFINITELY Corrupt data for image id={id}, openImageId={tag}, path={path} --> CHECK MANUALLY!".format(
                            id=sample.id, tag=tag, path=path
                        )
                    )
            if p.progress > 0 and ((cc % p.progress) == 0):
                print("sample: ", cc, "/", len(dataset), "tag:", tag)
            if p.progressbar:
                pb.update()

    print("\nHAVE A NICE DAY!\n")
