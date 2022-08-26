import numpy as np
import logging, io, shlex, shutil
import os, time, glob
import subprocess
from hashlib import md5

from .base import EncoderDecoder
from PIL import Image


def test_command(comm):
    try:
        p=subprocess.Popen(comm, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    except FileNotFoundError:
        raise
    else:
        return comm


class VTMEncoderDecoder(EncoderDecoder):
    """EncoderDecoder class for VTM encoder

    :param encoderApp: VTM encoder command
    :param decoderApp: VTM decoder command
    :param vtm_cfg: path of encoder cfg file
    :param ffmpeg: ffmpeg command
    :param qp: the default quantization parameter of the instance. Integer from 0 to 63.  Default=30.
    :param save: save intermediate steps into member ``saved`` (for debugging).

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
        bpp, img_hat = encdec.BGR(cv2.imread("fname.png"))

    You can enable caching and avoid re-encoding of images like this:

    ::

        encdec=VTMEncoderDecoder(encoderApp=encoderApp, decoderApp=decoderApp, ffmpeg="ffmpeg", vtm_cfg=getDataFile("encoder_intra_vtm_1.cfg"), qp=47, cache="/tmp/kokkelis")
        bpp, img_hat = encdec.BGR(cv2.imread("fname.png"), tag="a_unique_tag")

    Cache can be inspected with:

    ::

        encdec.dump()


    """
    def __init__(self, encoderApp=None, decoderApp=None, ffmpeg="ffmpeg", vtm_cfg=None, qp=30, save=False, base_path="/dev/shm", cache=None):
        self.logger = logging.getLogger(self.__class__.__name__)
        assert(encoderApp is not None), "please give encoder command"
        assert(decoderApp is not None), "please give decoder command"
        assert(vtm_cfg is not None), "please give VTM config file"
        # test commands
        self.encoderApp=test_command(encoderApp)
        self.decoderApp=test_command(decoderApp)
        self.ffmpeg=test_command(ffmpeg)
        assert(os.path.isfile(vtm_cfg)), "can't find "+vtm_cfg
        assert(os.path.isdir(base_path)), "can't find "+base_path

        self.encoderApp=encoderApp
        self.decoderApp=decoderApp
        self.ffmpeg=ffmpeg
        self.vtm_cfg = vtm_cfg
        self.qp=qp
        self.save=save
        self.base_path=base_path
        
        if cache is not None:
            if not os.path.isdir(cache):
                self.logger.info("creating %s", cache)
                os.makedirs(cache)
            self.folder=cache
            self.caching=True
        else:
            self.caching=False
            self.folder = os.path.join(self.base_path, "vtm_" + str(id(self)))

        try:
            os.mkdir(self.folder)
        except FileExistsError:
            assert(os.path.isdir(self.folder))
            self.logger.warning("folder %s exists already", self.folder)
        self.reset()

    def __str__(self):
        st=""
        st += "encoderApp:   "+self.encoderApp + "\n"
        st += "decoderApp:   "+self.decoderApp + "\n"
        st += "ffmpeg    :   "+self.ffmpeg + "\n"
        st += "qp        :   "+str(self.qp) + "\n"
        st += "path      :   "+self.folder + "\n"
        if self.caching:
            st += "CACHING ENABLED\n"
        return st

    def dump(self):
        """Dumps files cached on disk by the VTMEncoderDecoder
        """
        print("contents of", self.folder)
        for fname in glob.glob(os.path.join(self.folder, "*")):
            print("    ", fname)

    def __del__(self):
        if self.caching:
            return
        # print("VTM: __del__", len(glob.glob(os.path.join(self.folder,"*"))))
        if len(glob.glob(os.path.join(self.folder,"*"))) > 5:
            # add some security here if user fat-fingers self.base_bath --> self.folder
            self.logger.critical("there are multiple files in %s : please remove manually", self.folder)
            return
        # print("removing", self.folder)
        self.logger.debug("removing %s", self.folder)
        shutil.rmtree(self.folder)

    def reset(self):
        """Reset encoder/decoder internal state?  Is there any?
        """
        super().reset()
        self.saved={}
        self.imcount=0

    def __ff_op__(self, rgb_image:np.array, op) -> np.array:
        """takes as an input a numpy RGB array (y,x,3)
        
        Outputs numpy RGB array after certain transformation
        """
        pil_img=Image.fromarray(rgb_image)
        f = io.BytesIO()
        pil_img.save(f, format="png")
        
        comm='{ffmpeg} -y -hide_banner -loglevel error -i pipe: -vf "{op}" -f apng pipe:'.format(
            ffmpeg=self.ffmpeg,
            op=op)
        self.logger.debug(comm)
        args = shlex.split(comm)
        p=subprocess.Popen(args, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        stdout, stderr=p.communicate(f.getvalue())
        if (stdout is None) or (len(stdout)<5): # say
            # print(stderr.decode("utf-8"))
            self.logger.fatal(stderr.decode("utf-8"))
            return None
        f2=io.BytesIO(stdout)
        pil_img2=Image.open(f2).convert("RGB")
        return np.array(pil_img2)


    def __ff_RGB24ToRAW__(self, rgb_image:np.array, form) -> bytes:
        """takes as an input a numpy RGB array (y,x,3)
        
        ffmpeg -i input.png -f rawvideo -pix_fmt yuv420p -dst_range 1 output.yuv
        
        produces raw video frame bytes in the given pixel format
        """
        pil_img=Image.fromarray(rgb_image)
        f = io.BytesIO()
        pil_img.save(f, format="png")
        
        comm='{ffmpeg} -y -hide_banner -loglevel error -i pipe: -f rawvideo -pix_fmt {form} -dst_range 1 pipe:'.format(
            ffmpeg=self.ffmpeg,
            form=form)
        self.logger.debug(comm)
        args = shlex.split(comm)
        p=subprocess.Popen(args, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        stdout, stderr=p.communicate(f.getvalue())
        if (stdout is None) or (len(stdout)<5): # say
            #print("ERROR")
            #print(stderr.decode("utf-8"))
            self.logger.fatal(stderr.decode("utf-8"))
            return None
        f2=io.BytesIO(stdout)
        return f2.read()


    def __ff_RAWToRGB24__(self, raw:bytes, form, width=None, height=None) -> bytes:
        """takes as an input a numpy RGB array (y,x,3)
        
        ffmpeg -y -f rawvideo -pix_fmt yuv420p -s 768x512 -src_range 1 -i test.yuv -frames 1 -pix_fmt rgb24 out.png
        
        produces RGB image from raw video format
        """
        assert(width is not None)
        assert(height is not None)
        assert(isinstance(raw, bytes))
        
        comm='{ffmpeg} -y -hide_banner -loglevel error -f rawvideo -pix_fmt {form} -s {width}x{height} -src_range 1 -i pipe: -frames 1 -pix_fmt rgb24 -f apng pipe:'.format(
            ffmpeg=self.ffmpeg, form=form, width=width, height=height)
        self.logger.debug(comm)
        args = shlex.split(comm)
        p=subprocess.Popen(args, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        stdout, stderr=p.communicate(raw)
        if (stdout is None) or (len(stdout)<5): # say
            #print("ERROR")
            # print(stderr.decode("utf-8"))
            self.logger.fatal(stderr.decode("utf-8"))
            return None
        f=io.BytesIO(stdout)
        pil_img=Image.open(f)
        return np.array(pil_img)


    def __VTMEncode__(self, inp_yuv_path=None, bin_path=None, width=None, height=None):
        assert(inp_yuv_path is not None)
        assert(bin_path is not None)
        assert(width is not None)
        assert(height is not None)
        comm='{encoderApp} -c {vtm_cfg} -i {inp_yuv_path} -b {bin_path} -o {out_yuv_path} -fr 1 -f 1 -wdt {wdt} -hgt {hgt} -q {qp} --ConformanceWindowMode=1 --InternalBitDepth=10'.format(
            encoderApp=self.encoderApp,
            vtm_cfg=self.vtm_cfg,
            inp_yuv_path=inp_yuv_path, # IN
            out_yuv_path="nada.yuv", # OUT # NOT USED
            bin_path=bin_path, # OUT
            wdt=width,
            hgt=height,
            qp=self.qp,
        )
        self.logger.debug(comm)
        args = shlex.split(comm)
        p=subprocess.Popen(args, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        stdout, stderr=p.communicate()


    def __VTMDecode__(self, bin_path=None, rec_yuv_path=None):
        assert(bin_path is not None)
        assert(rec_yuv_path is not None)
        comm='{decoderApp} -b {bin_path} -o {rec_yuv_path}'.format(
            decoderApp=self.decoderApp,
            bin_path=bin_path, # IN
            rec_yuv_path=rec_yuv_path # OUT
        )
        self.logger.debug(comm)
        args = shlex.split(comm)
        p=subprocess.Popen(args, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        stdout, stderr=p.communicate()


    def BGR(self, bgr_image, tag=None):
        """
        :param bgr_image: numpy BGR image (y,x,3)
        :param tag: a string that can be used to identify & cache images (optional)


        Returns BGR image that has gone through VTM encoding and decoding process and all other operations as defined by MPEG VCM, namely:

        ::

            padded_hgt = math.ceil(height/2)*2
            padded_wdt = math.ceil(width/2)*2
            1. ffmpeg -i {input_jpg_path} -vf “pad=ceil(iw/2)*2:ceil(ih/2)*2” {input_tmp_path}
            2. ffmpeg -i {input_tmp_path} -f rawvideo -pix_fmt yuv420p -dst_range 1 {yuv_image_path}
            3. {VTM_encoder_path} -c {VTM_AI_cfg} -i {yuv_image_path} -b {bin_image_path} -o {temp_yuv_path} -fr 1 -f 1 -wdt {padded_wdt} -hgt {padded_hgt} -q {qp} --ConformanceWindowMode=1 --InternalBitDepth=10
            4. {VTM_decoder_path} -b {bin_image_path} -o {rec_yuv_path}
            5. ffmpeg -y -f rawvideo -pix_fmt yuv420p10le -s {padded_wdt}x{padded_hgt} -src_range 1 -i {rec_yuv_path} -frames 1  -pix_fmt rgb24 {rec_png_path}
            6. ffmpeg -y -i {rec_png_path} -vf "crop={width}:{height}" {rec_image_path}


        """
        ## we could use this to create unique filename if we want cache & later identify the images:
        #"X".join([str(n) for n in md5(bgr_image).digest()])
        ##but it's better to use explicit tags as provided by the user
        fname_yuv=os.path.join(self.folder, "tmp.yuv") # yuv produced by ffmpeg
        if self.caching:
            assert(tag is not None), "caching requested, but got no tag"
            fname_bin=os.path.join(self.folder, "bin_"+tag) # bin produced by VTM
        else:
            fname_bin=os.path.join(self.folder, "bin") # bin produced by VTM
        fname_rec=os.path.join(self.folder, "rec.yuv") # yuv produced by VTM

        rgb_image = bgr_image[:,:,[2,1,0]] # BGR --> RGB
        # apply ffmpeg commands as defined in MPEG VCM group docs
        # each submethod should cite the correct command

        # 1. MPEG-VCM: ffmpeg -i {input_jpg_path} -vf “pad=ceil(iw/2)*2:ceil(ih/2)*2” {input_tmp_path}
        padded=self.__ff_op__(rgb_image, "pad=ceil(iw/2)*2:ceil(ih/2)*2")

        if (not self.caching) or (not os.path.isfile(fname_bin)):
            
            # 2. MPEG-VCM: ffmpeg -i {input_tmp_path} -f rawvideo -pix_fmt yuv420p -dst_range 1 {yuv_image_path}
            yuv_bytes=self.__ff_RGB24ToRAW__(padded, "yuv420p")

            ## this is not needed since each VTMEncoderDecoder has its own directory
            # tmu=int(time.time()*1E6) # microsec timestamp
            # fname=os.path.join(self.folder, str(tmu))
            ## ..you could also use the tag to cache the encoded images if you'd like to do caching
            
            self.logger.debug("writing %s", fname_yuv)
            with open(fname_yuv, "wb") as f:
                f.write(yuv_bytes)

            # 3. MPEG-VCM: {VTM_encoder_path} -c {VTM_AI_cfg} -i {yuv_image_path} -b {bin_image_path} 
            #               -o {temp_yuv_path} -fr 1 -f 1 -wdt {padded_wdt} -hgt {padded_hgt} -q {qp} --ConformanceWindowMode=1 --InternalBitDepth=10
            self.__VTMEncode__(inp_yuv_path=fname_yuv, bin_path=fname_bin, width=padded.shape[1], height=padded.shape[0])
            self.logger.debug("removing %s", fname_yuv)
            os.remove(fname_yuv) # cleanup

        # calculate bpp
        self.logger.debug("reading %s for bpp calculation", fname_bin)
        with open(fname_bin, "rb") as f:
            n_bytes=len(f.read())
        bpp=n_bytes*8/(rgb_image.shape[1]*rgb_image.shape[0])

        # 4. MPEG-VCM: {VTM_decoder_path} -b {bin_image_path} -o {rec_yuv_path}
        self.__VTMDecode__(bin_path=fname_bin, rec_yuv_path=fname_rec)

        self.logger.debug("reading %s", fname_rec)
        with open(fname_rec, "rb") as f:
            yuv_bytes_hat=f.read()
        self.logger.debug("removing %s", fname_rec)
        os.remove(fname_rec) # cleanup

        if not self.caching:
            self.logger.debug("removing %s", fname_bin)
            os.remove(fname_bin)

        # 5. MPEG-VCM: ffmpeg -y -f rawvideo -pix_fmt yuv420p10le -s {padded_wdt}x{padded_hgt} -src_range 1 -i {rec_yuv_path} -frames 1  -pix_fmt rgb24 {rec_png_path}
        form="yuv420p10le"
        padded_hat=self.__ff_RAWToRGB24__(yuv_bytes_hat, form=form, width=padded.shape[1],
            height=padded.shape[0])

        # 6. MPEG-VCM: ffmpeg -y -i {rec_png_path} -vf "crop={width}:{height}" {rec_image_path}
        rgb_image_hat=self.__ff_op__(padded_hat, "crop={width}:{height}".format(
            width=rgb_image.shape[1],
            height=rgb_image.shape[0]
        ))
        if self.save:
            self.saved={
                "rgb_image"     : rgb_image,
                "padded"        : padded,
                "padded_hat"    : padded_hat,
                "rgb_image_hat" : rgb_image_hat
            }
        else:
            self.saved = {}

        bgr_image_hat=rgb_image_hat[:,:,[2,1,0]] # RGB --> BGR
        self.logger.debug("input & output sizes: %s %s. bps = %s", bgr_image.shape, bgr_image_hat.shape, bpp)
        return bpp, bgr_image_hat
