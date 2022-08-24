from cv2 import transform
from matplotlib.transforms import Transform
import numpy as np
import logging, io, shlex
import os
import math
import subprocess
from regex import B

from .base import EncoderDecoder
from PIL import Image


class VTMEncoderDecoder(EncoderDecoder):
    """EncoderDecoder class for VTM encoder

    :param encoderApp: path of VTM encoder
    :param decoderApp: path of VTM decoder
    :param vtm_cfg: path of encoder cfg
    :param qp: the default quantization parameter of the instance. Integer from 0 to 63
    :param save_transformed: option to save intermedidate images. default = False
    """
    def __init__(self, encoderApp="foo", decoderApp="foo", ffmpeg="ffmpeg", vtm_cfg, qp, save_transformed=False):
        self.logger = logging.getLogger(self.__class__.__name__)
        # TODO: save vtm_cfg as a constant in python file or in data files
        if os.path.isfile(vtm_cfg):
            self.vtm_cfg = vtm_cfg
        if os.path.isfile(encoderApp):
            self.encoderApp = encoderApp
        else:
            self.logger.critical("VTM encoder not found at %s", encoderApp)
        if os.path.isfile(decoderApp):
            self.decoderApp = decoderApp
        else:
            self.logger.critical("VTM decoder not found at %s", decoderApp)
        self.tmp_output_folder = '/dev/shm/'
        self.qp = qp
        self.save_transformed = save_transformed
        self.ffmpeg = "ffmpeg"
        self.reset()

    def reset(self):
        """Reset encoder/decoder internal state? Jacky: TODO.
        """
        super().reset()
        self.imcount=0


    def ff_op(self, rgb_image:np.array, op) -> np.array:
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


    def ff_RGB24ToRAW(self, rgb_image:np.array, form) -> bytes:
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


    def ff_RAWToRGB24(self, raw:bytes, form, width=None, height=None) -> bytes:
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


    def process_cmd(self, cmd, print_out=False):
        """
        process bash cmd
        :param cmd: bash command 
        :param print_out: show printout. Default: False
        """
        
        if print_out:
            print(cmd)
        p = subprocess.Popen(cmd, shell=True, stderr=subprocess.STDOUT, stdout=subprocess.PIPE)
        
        for line in p.stdout.readlines():
            if print_out:
                self.logger.debug(line)
                print(line)
        
        if print_out:
            print('Done')

        """
        # use VTM encoder to generate bitstream
        cmd_str= f'{self.encoderApp} -c {self.vtm_cfg} -i {yuv_image_path} -b {bin_path} -o {temp_yuv_path} -fr 1 -f 1 -wdt {padded_wdt} -hgt {padded_hgt} -q {qp}'
        self.process_cmd(cmd_str, print_out= bPrint)
        # check if bin is generated correctly
        check_bin = os.path.isfile(bin_path)
        if check_bin:
            self.logger.debug("Bitstream stored at: %s", bin_path)
        else:
            self.logger.critical("Bitstream %s storage failed.", bin_path)
    
        # use VTM decoder
        cmd_str= f'{self.decoderApp} -b {bin_path} -o {rec_yuv_path}'
        self.process_cmd(cmd_str, print_out = bPrint)
        """

    
    def BGR(self, bgr_image):
        """
        :param bgr_image: numpy BGR image (y,x,3)

        Returns BGR image that has gone through VTM encode/decode.  Jacky: TODO

        ::

            padded_hgt = math.ceil(height/2)*2
            padded_wdt = math.ceil(width/2)*2
            ffmpeg -i {input_jpg_path} -vf “pad=ceil(iw/2)*2:ceil(ih/2)*2” {input_tmp_path}
            ffmpeg -i {input_tmp_path} -f rawvideo -pix_fmt yuv420p -dst_range 1 {yuv_image_path}
            {VTM_encoder_path} -c {VTM_AI_cfg} -i {yuv_image_path} -b {bin_image_path} -o {temp_yuv_path} -fr 1 -f 1 -wdt {padded_wdt} -hgt {padded_hgt} -q {qp} --ConformanceWindowMode=1 --InternalBitDepth=10
            {VTM_decoder_path} -b {bin_image_path} -o {rec_yuv_path}
            ffmpeg -y -f rawvideo -pix_fmt yuv420p10le -s {padded_wdt}x{padded_hgt} -src_range 1 -i {rec_yuv_path} -frames 1  -pix_fmt rgb24 {rec_png_path}
            ffmpeg -y -i {rec_png_path} -vf "crop={width}:{height}" {rec_image_path}


        """
        rgb_image = bgr_image[:,:,[2,1,0]] # BGR --> RGB
        # apply ffmpeg commands as defined in MPEG VCM group docs
        # each submethod should cite the correct command

        # MPEG-VCM: ffmpeg -i {input_jpg_path} -vf “pad=ceil(iw/2)*2:ceil(ih/2)*2” {input_tmp_path}
        padded=self.ff_op(rgb_image, "pad=ceil(iw/2)*2:ceil(ih/2)*2")

        # MPEG-VCM: ffmpeg -i {input_tmp_path} -f rawvideo -pix_fmt yuv420p -dst_range 1 {yuv_image_path}
        yuv_bytes=self.ff_RGB24ToRAW(padded)

        # TODO: dump bytes to disk & call VTM Encoder
        # TODO: call VTM Decoder & read bytes from disk
        # TODO: time VTM encoder/decode, calculate bpp

        # MPEG-VCM: ffmpeg -y -f rawvideo -pix_fmt yuv420p10le -s {padded_wdt}x{padded_hgt} -src_range 1 -i {rec_yuv_path} -frames 1  -pix_fmt rgb24 {rec_png_path}
        padded_hat=self.ff_RAWToRGB24(yuv_bytes_hat, form="yuv420p10le", width=padded.shape[1],
            height=padded.shape[0])

        # MPEG-VCM: ffmpeg -y -i {rec_png_path} -vf "crop={width}:{height}" {rec_image_path}
        rgb_image_hat=self.ff_op(padded_hat, "crop={width}:{height}".format(
            width=rgb_image.shape[1],
            height=rgb_image.shape[0]
        )

        if self.save:
            self.saved={
                "rgb_image"     : rgb_image,
                "padded "       : padded,
                "padded_hat"    : padded_hat,
                "rgb_image_hat" : rgb_image_hat
            }
        else:
            self.saved = None

        bgr_image_hat=rgb_image_hat[:,:,[2,1,0]] # RGB --> BGR
        self.logger.debug("input & output sizes: %s %s. bps = %s", bgr_image.shape, bgr_image_hat.shape, bpp[0])
        # print(">> cc, bpp_sum ", self.cc, self.bpp_sum)
        return bpp[0], bgr_image_hat
