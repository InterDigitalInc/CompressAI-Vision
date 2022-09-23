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

import glob
import io
import logging
import os
import shlex
import shutil
import subprocess
from uuid import uuid4 as uuid

import numpy as np

from PIL import Image

from .base import EncoderDecoder

from compressai_vision.ffmpeg import FFMpeg
from compressai_vision.tools import test_command, dumpImageArray
from compressai_vision.constant import vf_per_scale


def removeFileIf(path) -> bool:
    try:
        os.remove(path)
    except FileNotFoundError:
        return False
    else:
        return True


class VTMEncoderDecoder(EncoderDecoder):
    """EncoderDecoder class for VTM encoder

    :param encoderApp: VTM encoder command
    :param decoderApp: VTM decoder command
    :param vtm_cfg: path of encoder cfg file
    :param ffmpeg: ffmpeg command used for padding/scaling
    :param qp: the default quantization parameter of the instance. Integer from 0 to 63.  Default=30.
    :param scale: enable the VCM working group defined padding/scaling pre & post-processings steps.
                  Possible values: 100 (default), 75, 50, 25.  Special value: None = ffmpeg scaling.  100 equals to a simple padding operation
    :param save: save intermediate steps into member ``saved`` (for debugging). Default: False.
    :param cache: (optional) define a directory where all encoded bitstreams are cached.
                  NOTE: If scale is defined, "scale/qp/" is appended to the cache path.  If no scale is defined, the appended path is "0/qp/"
    :param dump: debugging option: dump input, intermediate and output images to disk in local directory
    :param skip: if bitstream is found in cache, then do absolutely nothing.  Good for restarting the bitstream generation. default: False.
                 When enabled, method BGR returns (0, None).  NOTE: do not use if you want to verify the bitstream files.
    :param warn: warn always when a bitstream is generated.  default: False.

    This class tries always to use the cached bitstreams if they are available (for this you need to define a cache directory, see above).  If the bitstream
    is available in cache, it will be used and the encoding step is skipped.  Otherwise encoder is started to produce bitstream.

    Example:

    ::

        import cv2, os, logging
        from compressai_vision.evaluation.pipeline import VTMEncoderDecoder
        from compressai_vision.tools import getDataFile

        path="/path/to/VVCSoftware_VTM/bin"
        encoderApp=os.path.join(path, "EncoderAppStatic")
        decoderApp=os.path.join(path, "DecoderAppStatic")

        # enable debugging log to see explicitly all the steps
        loglev=logging.DEBUG
        quickLog("VTMEncoderDecoder", loglev)

        encdec=VTMEncoderDecoder(encoderApp=encoderApp, decoderApp=decoderApp, ffmpeg="ffmpeg", vtm_cfg=getDataFile("encoder_intra_vtm_1.cfg"), qp=47)
        nbits, img_hat = encdec.BGR(cv2.imread("fname.png"))

    You can enable caching and avoid re-encoding of images:

    ::

        encdec=VTMEncoderDecoder(encoderApp=encoderApp, decoderApp=decoderApp, ffmpeg="ffmpeg", vtm_cfg=getDataFile("encoder_intra_vtm_1.cfg"), qp=47, cache="/tmp/kokkelis")
        nbits, img_hat = encdec.BGR(cv2.imread("fname.png"), tag="a_unique_tag")

    Cache can be inspected with:

    ::

        encdec.dump()


    """

    def __init__(
        self,
        encoderApp=None,
        decoderApp=None,
        ffmpeg="ffmpeg",
        vtm_cfg=None,
        qp=47,
        scale=100,
        save=False,
        base_path="/dev/shm",
        cache=None,
        dump=False,
        skip=False,
        keep=False,
        warn=False,
    ):
        self.logger = logging.getLogger(self.__class__.__name__)
        assert encoderApp is not None, "please give encoder command"
        assert decoderApp is not None, "please give decoder command"
        assert vtm_cfg is not None, "please give VTM config file"

        self.scale = scale
        if self.scale is not None:
            assert self.scale in vf_per_scale.keys(), "wrong scaling factor"

        self.vtm_cfg = vtm_cfg
        self.qp = qp
        self.save = save
        self.base_path = base_path
        self.caching = False
        self.dump = dump
        self.skip = skip
        self.keep = keep
        self.warn = warn

        self.save_folder = "vtm_encoder_decoder"
        if self.dump:
            self.logger.warning(
                "Will save intermediate images to local folder %s", self.save_folder
            )
            os.makedirs(self.save_folder, exist_ok=True)

        if cache is not None:
            if not os.path.isdir(cache):
                self.logger.info("creating %s", cache)
                os.makedirs(cache)
            # let's make the life easier for the user
            # for caching, they won't remember to include the quality parameter
            # value into the path anyway (so that files corresponding to different qps don't get mixed up)
            # so we'll do it here:
            if scale is None:
                self.folder = os.path.join(cache, "0", str(self.qp))
            else:
                self.folder = os.path.join(cache, str(self.scale), str(self.qp))
            self.caching = True
        else:
            self.caching = False
            # uid=str(id(self))
            uid = str(uuid())  # safer
            self.folder = os.path.join(self.base_path, "vtm_" + uid)

        # test commands
        self.encoderApp = test_command(encoderApp)
        self.decoderApp = test_command(decoderApp)
        try:
            self.ffmpeg_comm = test_command(ffmpeg)
        except FileNotFoundError:
            raise (AssertionError("cant find ffmpeg"))
        assert os.path.isfile(vtm_cfg), "can't find " + vtm_cfg
        assert os.path.isdir(base_path), "can't find " + base_path

        # self.encoderApp = encoderApp
        # self.decoderApp = decoderApp
        # self.ffmpeg = ffmpeg
        self.ffmpeg = FFMpeg(self.ffmpeg_comm, self.logger)

        try:
            os.makedirs(self.folder, exist_ok=False)
        except FileExistsError:
            assert os.path.isdir(self.folder)
            self.logger.warning("folder %s exists already", self.folder)
        self.reset()

    def __str__(self):
        st = ""
        st += "encoderApp:   " + self.encoderApp + "\n"
        st += "decoderApp:   " + self.decoderApp + "\n"
        st += "ffmpeg    :   " + self.ffmpeg + "\n"
        st += "qp        :   " + str(self.qp) + "\n"
        st += "path      :   " + self.folder + "\n"
        if self.caching:
            st += "CACHING ENABLED\n"
        return st

    def dump(self):
        """Dumps files cached on disk by the VTMEncoderDecoder"""
        print("contents of", self.folder)
        for fname in glob.glob(os.path.join(self.folder, "*")):
            print("    ", fname)

    def getCacheDir(self):
        """Returns directory where temporary and cached files are saved"""
        return self.folder

    def __del__(self):
        if not hasattr(self, "caching"):
            return  # means ctor crashed
        if self.keep:
            return
        if self.caching:
            return
        # print("VTM: __del__", len(glob.glob(os.path.join(self.folder,"*"))))
        if len(glob.glob(os.path.join(self.folder, "*"))) > 5:
            # add some security here if user fat-fingers self.base_bath --> self.folder
            self.logger.critical(
                "there are multiple files in %s : please remove manually", self.folder
            )
            return
        # print("removing", self.folder)
        if True:
            # if False:
            self.logger.debug("removing %s", self.folder)
            shutil.rmtree(self.folder)

    def reset(self):
        """Reset encoder/decoder internal state.  At the moment, there ain't any."""
        super().reset()
        self.saved = {}
        self.imcount = 0

    def __VTMEncode__(
        self,
        inp_yuv_path=None,
        out_yuv_path=None,
        bin_path=None,
        width=None,
        height=None,
    ) -> bool:
        assert inp_yuv_path is not None
        assert out_yuv_path is not None
        assert bin_path is not None
        assert width is not None
        assert height is not None
        comm = "{encoderApp} -c {vtm_cfg} -i {inp_yuv_path} -b {bin_path} -o {out_yuv_path} -fr 1 -f 1 -wdt {wdt} -hgt {hgt} -q {qp} --ConformanceWindowMode=1 --InternalBitDepth=10".format(
            encoderApp=self.encoderApp,
            vtm_cfg=self.vtm_cfg,
            inp_yuv_path=inp_yuv_path,  # IN
            out_yuv_path=out_yuv_path,  # OUT # NOT USED
            bin_path=bin_path,  # OUT
            wdt=width,
            hgt=height,
            qp=self.qp,
        )
        self.logger.debug(comm)
        args = shlex.split(comm)
        p = subprocess.Popen(
            args, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )
        stdout, stderr = p.communicate()
        if p.returncode != 0:
            """
            raise (
                AssertionError(
                    "VTM encode failed with:\n"
                    + stderr.decode("utf-8")
                    + "\nYOU PROBABLY SHOULD ENABLE FFMPEG SCALING\n"
                )
            )
            """
            self.logger.fatal("VTM encode failed with %s", stderr.decode("utf-8"))
            self.logger.fatal("\nYOU PROBABLY SHOULD ENABLE FFMPEG SCALING\n")
            return False
        else:
            return True

    def __VTMDecode__(self, bin_path=None, rec_yuv_path=None) -> bool:
        assert bin_path is not None
        assert rec_yuv_path is not None
        comm = "{decoderApp} -b {bin_path} -o {rec_yuv_path}".format(
            decoderApp=self.decoderApp,
            bin_path=bin_path,  # IN
            rec_yuv_path=rec_yuv_path,  # OUT
        )
        self.logger.debug(comm)
        args = shlex.split(comm)
        p = subprocess.Popen(
            args, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )
        stdout, stderr = p.communicate()
        if p.returncode != 0:
            # raise (AssertionError("VTM decode failed with " + stderr.decode("utf-8")))
            self.logger.fatal("VTM encode failed with %s", stderr.decode("utf-8"))
            return False
        else:
            return True

    def BGR(self, bgr_image, tag=None) -> tuple:
        """
        :param bgr_image: numpy BGR image (y,x,3)
        :param tag: a string that can be used to identify & cache images (optional).  Necessary if you're using caching

        Returns BGR image that has gone through VTM encoding and decoding process and all other operations as defined by MPEG VCM.

        Returns a tuple of (nbits, transformed_bgr_image)

        This method is somewhat complex: in addition to perform the necessary image transformation, it also handles caching of bitstreams,
        inspection if bitstreams exist, etc.  Error conditions from ffmpeg and/or from VTMEncoder/Decoder must be taken correctly into account.

        VCM working group ops:

        ::

            padded_hgt = math.ceil(height/2)*2
            padded_wdt = math.ceil(width/2)*2
            1. ffmpeg vf -i {input_tmp_path} -o {input_padded_tmp_path}

            vf depends on the scale:

            for 100%: -vf “pad=ceil(iw/2)*2:ceil(ih/2)*2”           # NOTE: simply padding
            for 75%:  -vf "scale=ceil(iw*3/8)*2:ceil(ih*3/8)*2"
            for 50%:  -vf "scale=ceil(iw/4)*2:ceil(ih/4)*2"
            for 25%:  -vf "scale=ceil(iw/8)*2:ceil(ih/8)*2"

            2. ffmpeg -i {input_padded_tmp_path} -f rawvideo -pix_fmt yuv420p -dst_range 1 {yuv_image_path}
            3. {VTM_encoder_path} -c {VTM_AI_cfg} -i {yuv_image_path} -b {bin_image_path} -o {temp_yuv_path} -fr 1 -f 1 -wdt {padded_wdt} -hgt {padded_hgt}
                -q {qp} --ConformanceWindowMode=1 --InternalBitDepth=10
            4. {VTM_decoder_path} -b {bin_image_path} -o {rec_yuv_path}
            5. ffmpeg -y -f rawvideo -pix_fmt yuv420p10le -s {padded_wdt}x{padded_hgt} -src_range 1 -i {rec_yuv_path} -frames 1 -pix_fmt rgb24 {rec_png_path}
            6. ffmpeg -y -i {rec_png_path} -vf "crop={width}:{height}" {rec_image_path} # NOTE: This can be done only if scale=100%, i.e. to remove padding
        """
        # we could use this to create unique filename if we want cache & later identify the images:
        # "X".join([str(n) for n in md5(bgr_image).digest()])
        # but it's better to use explicit tags as provided by the user

        if self.caching:
            assert tag is not None, "caching requested, but got no tag"
            fname_bin = os.path.join(self.folder, "bin_" + tag)  # bin produced by VTM
        else:
            # if no caching, we have a unique directory where all this stuff goes, so no need to separate the files
            # with uuids
            tag = ""
            fname_bin = os.path.join(self.folder, "bin")  # bin produced by VTM

        if self.skip:
            assert self.caching, "skip requires caching enabled"

        """A separate checkmode is not a good idea.. either check if the file exists (quickcheck) or otherwise
        do the whole pipeline (using the existing bitstream)

        if self.caching and self.checkmode:
            self.logger.debug("checkmode: looking for file %s", fname_bin)
            # just check if required bitstream exists.  return 0 if ok, -1 if not there
            if os.path.isfile(fname_bin):
                self.logger.debug("checkmode: test reading file %s", fname_bin)
                with open(fname_bin, "rb") as f:
                    bitstream = f.read()
                if len(bitstream) < 1:
                    self.logger.warning("checkmode: found empty file for %s: will remove", fname_bin)
                    removeFileIf(fname_bin)
                # cached bitstream exists allright
                return 0, None
            else:
                self.logger.debug("Checkmode: %s does not exist", fname_bin)
                return -1, None
        """
        if self.skip:
            if os.path.isfile(fname_bin) and (os.path.getsize(fname_bin) > 5):
                self.logger.debug(
                    "Found file %s from cache & skip enabled: returning 0, None",
                    fname_bin,
                )
                return 0, None
            else:
                self.logger.debug(
                    "Couldn't find file %s from cache (or its zero-length) & skip enabled: returning -1, None",
                    fname_bin,
                )
                return -1, None

        # uid=str(uuid())
        uid = tag  # the tag is supposedly unique, so use that to mark all files
        fname_yuv = os.path.join(
            self.folder, "tmp_%s.yuv" % (uid)
        )  # yuv produced by ffmpeg
        fname_yuv_out = os.path.join(
            self.folder, "nada_%s.yuv" % (uid)
        )  # yuv produced VTM.. not used

        fname_rec = os.path.join(
            self.folder, "rec_%s.yuv" % (uid)
        )  # yuv produced by VTM

        rgb_image = bgr_image[:, :, [2, 1, 0]]  # BGR --> RGB
        # apply ffmpeg commands as defined in MPEG VCM group docs
        # each submethod should cite the correct command

        if self.dump:
            dumpImageArray(
                rgb_image, self.save_folder, "original_" + str(self.imcount) + ".png"
            )

        if self.scale is not None:
            # 1. MPEG-VCM: ffmpeg -i {input_jpg_path} -vf “pad=ceil(iw/2)*2:ceil(ih/2)*2” {input_tmp_path}
            vf = vf_per_scale[self.scale]
            padded = self.ffmpeg.ff_op(rgb_image, vf)
            if padded is None:
                self.logger.fatal(
                    "ffmpeg scale operation failed: will skip image %s", tag
                )
                return -1, None
            if self.dump:
                dumpImageArray(
                    padded,
                    self.save_folder,
                    "ffmpeg_scaled_" + str(self.imcount) + ".png",
                )
        else:
            padded = rgb_image

        if (not self.caching) or (not os.path.isfile(fname_bin)):
            self.logger.debug("Creating file %s with ffmpeg", fname_yuv)
            # 2. MPEG-VCM: ffmpeg -i {input_tmp_path} -f rawvideo -pix_fmt yuv420p -dst_range 1 {yuv_image_path}
            yuv_bytes = self.ffmpeg.ff_RGB24ToRAW(padded, "yuv420p")
            if yuv_bytes is None:
                self.logger.fatal(
                    "ffmpeg to yuv conversion failed: will skip image %s", tag
                )
                return -1, None

            # this is not needed since each VTMEncoderDecoder has its own directory
            # tmu=int(time.time()*1E6) # microsec timestamp
            # fname=os.path.join(self.folder, str(tmu))
            # ..you could also use the tag to cache the encoded images if you'd like to do caching
            self.logger.debug(
                "writing %s output from ffmpeg to disk (for VTMEncode to read it)",
                fname_yuv,
            )
            with open(fname_yuv, "wb") as f:
                f.write(yuv_bytes)

            # 3. MPEG-VCM: {VTM_encoder_path} -c {VTM_AI_cfg} -i {yuv_image_path} -b {bin_image_path}
            #               -o {temp_yuv_path} -fr 1 -f 1 -wdt {padded_wdt} -hgt {padded_hgt} -q {qp} --ConformanceWindowMode=1 --InternalBitDepth=10
            if self.warn:
                self.logger.warning(
                    "creating bitstream %s with VTMEncode from scratch", fname_bin
                )
            else:
                self.logger.debug(
                    "creating bitstream %s with VTMEncode from scratch", fname_bin
                )

            ok = self.__VTMEncode__(
                inp_yuv_path=fname_yuv,
                out_yuv_path=fname_yuv_out,
                bin_path=fname_bin,
                width=padded.shape[1],
                height=padded.shape[0],
            )
            # cleanup
            if not self.keep:
                self.logger.debug("removing %s from ffmpeg", fname_yuv)
                removeFileIf(fname_yuv)  # cleanup
                self.logger.debug("removing %s from VTMEncode", fname_yuv_out)
                removeFileIf(fname_yuv_out)  # cleanup

            if (not ok) or (not os.path.isfile(fname_bin)):
                self.logger.fatal("VTMEncode failed: will skip image %s", tag)
                return -1, None

        else:
            self.logger.debug("Using existing file %s from cache", fname_bin)

        # calculate nbits
        self.logger.debug("reading %s from VTMEncode", fname_bin)
        with open(fname_bin, "rb") as f:
            n_bytes = len(f.read())

        if n_bytes < 1:
            self.logger.fatal(
                "Empty output from VTMEncode: will skip image %s & remove the bitstream file",
                tag,
            )
            removeFileIf(fname_bin)
            return -1, None

        nbits = n_bytes * 8  # / (rgb_image.shape[1] * rgb_image.shape[0])

        # 4. MPEG-VCM: {VTM_decoder_path} -b {bin_image_path} -o {rec_yuv_path}
        ok = self.__VTMDecode__(bin_path=fname_bin, rec_yuv_path=fname_rec)

        if (not ok) or (not os.path.isfile(fname_rec)):
            self.logger.fatal(
                "VTMDecode failed: will skip image %s & remove the bitstream file", tag
            )
            removeFileIf(fname_rec)
            removeFileIf(fname_bin)
            return -1, None

        self.logger.debug("reading %s from VTMDecode", fname_rec)
        with open(fname_rec, "rb") as f:
            yuv_bytes_hat = f.read()

        if len(yuv_bytes_hat) < 1:
            self.logger.fatal(
                "Empty output from VTMDecode: will skip image %s & remove the bitstream file",
                tag,
            )
            removeFileIf(fname_rec)
            removeFileIf(fname_bin)
            return -1, None

        if not self.keep:
            self.logger.debug("removing %s from VTMDecode", fname_rec)
            removeFileIf(fname_rec)  # cleanup

        if not self.caching and not self.keep:
            self.logger.debug("removing %s from VTMEncode", fname_bin)
            removeFileIf(fname_bin)

        # 5. MPEG-VCM: ffmpeg -y -f rawvideo -pix_fmt yuv420p10le -s {padded_wdt}x{padded_hgt} -src_range 1 -i {rec_yuv_path} -frames 1  -pix_fmt rgb24 {rec_png_path}
        form = "yuv420p10le"
        padded_hat = self.ffmpeg.ff_RAWToRGB24(
            yuv_bytes_hat, form=form, width=padded.shape[1], height=padded.shape[0]
        )

        if padded_hat is None:
            self.logger.fatal(
                "ffmpeg raw->rgb24 operation failed: will skip image %s & remove bitstream file (if cached)",
                tag,
            )
            removeFileIf(fname_bin)
            return -1, None

        if self.scale is not None and self.scale == 100:
            # was scaled, so need to backscale
            # NOTE: this can only be done to the 100% "scaling" which is nothing else than just cropping
            # so we "backcrop" & remove the added borders
            # 6. MPEG-VCM: ffmpeg -y -i {rec_png_path} -vf "crop={width}:{height}" {rec_image_path}
            rgb_image_hat = self.ffmpeg.ff_op(
                padded_hat,
                "crop={width}:{height}".format(
                    width=rgb_image.shape[1], height=rgb_image.shape[0]
                ),
            )
            if rgb_image_hat is None:
                self.logger.fatal(
                    "ffmpeg crop operation failed: will skip image %s & remove bitstream file (if cached)",
                    tag,
                )
                removeFileIf(fname_bin)
                return -1, None
        else:
            rgb_image_hat = padded_hat

        if self.save:
            self.saved = {
                "rgb_image": rgb_image,
                "padded": padded,
                "padded_hat": padded_hat,
                "rgb_image_hat": rgb_image_hat,
            }
        else:
            self.saved = {}

        if self.dump:
            dumpImageArray(
                rgb_image_hat, self.save_folder, "final_" + str(self.imcount) + ".png"
            )

        bgr_image_hat = rgb_image_hat[:, :, [2, 1, 0]]  # RGB --> BGR
        self.logger.debug(
            "input & output sizes: %s %s. nbits = %s",
            bgr_image.shape,
            bgr_image_hat.shape,
            nbits,
        )
        self.imcount += 1
        return nbits, bgr_image_hat
