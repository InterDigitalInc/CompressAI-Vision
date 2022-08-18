from cv2 import transform
from matplotlib.transforms import Transform
import numpy as np
import logging
import os
import math
import subprocess
from regex import B

import torch
from .base import EncoderDecoder
from PIL import Image

from torchvision import transforms

class VTMEncoderDecoder(EncoderDecoder):
    """EncoderDecoder class for VTM encoder
    """


   
    def __init__(self, encoderApp, decoderApp, vtm_cfg, qp, save_transformed=False):
        self.logger = logging.getLogger(self.__class__.__name__)
        
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
        self.device = 'cpu'
        self.qp = qp
        self.save_transformed = save_transformed
        self.reset()

    def reset(self):
        super().reset()
        self.imcount=0


    def process_cmd(self, cmd, print_out=False):
        # cmd = f'ffmpeg -y -hide_banner -loglevel error -i {input_fname} -vf "crop={width}:{height}" {output_fname}'
        if print_out:
            print(cmd)
        p = subprocess.Popen(cmd, shell=True, stderr=subprocess.STDOUT, stdout=subprocess.PIPE)
        
        for line in p.stdout.readlines():
            if print_out:
                self.logger.debug(line)
                print(line)
        
        if print_out:
            print('Done')


    


    def __encode_ffmpeg__(self, x, qp, bin_path, bPrint=True):

        # self.logger.debug("feeding tensor to VTM encoder: shape: %s, type: %s", x.size, x.type)

# from Torch Tensor to Image   
        x=torch.squeeze(x, 0)
        tensor_size=x.size()

        image_name='tmp.jpg'

        width= tensor_size[-1]
        height= tensor_size[-2]
        padded_hgt = math.ceil(height/2)*2
        padded_wdt = math.ceil(width/2)*2

        
        
        tmp_jpg_path = os.path.join(self.tmp_output_folder, image_name)
        # some function to save x as tmp_jpg
        img= transforms.ToPILImage()(x).convert("RGB")
        img.save(tmp_jpg_path)


        yuv_image_path = os.path.join(self.tmp_output_folder, image_name.replace('.jpg', '.yuv'))
        temp_yuv_path = os.path.join(self.tmp_output_folder, 'rec.yuv')
        # 1. use ffmpeg to rescale and pad the input image, and then convert to yuv
        cmd_str= f'ffmpeg -y -i {tmp_jpg_path} -vf "pad=ceil(iw/2)*2:ceil(ih/2)*2" -f rawvideo -pix_fmt yuv420p -dst_range 1 {yuv_image_path}'
        self.process_cmd(cmd_str,print_out= bPrint)

        # 2. use VTM encoder to generate bitstream
        cmd_str= f'{self.encoderApp} -c {self.vtm_cfg} -i {yuv_image_path} -b {bin_path} -o {temp_yuv_path} -fr 1 -f 1 -wdt {padded_wdt} -hgt {padded_hgt} -q {qp}'
        self.process_cmd(cmd_str, print_out= bPrint)

        # 3. check if bin is generated correctly

        check_bin = os.path.isfile(bin_path)
        if check_bin:
            self.logger.debug("Bitstream stored at: %s", bin_path)

    def __decode_ffmpeg__(self, bin_path, width, height, bPrint=True):

        # self.logger.debug("feeding tensor to VTM encoder: shape: %s, type: %s", x.shape, x.type())
        padded_hgt = math.ceil(height/2)*2
        padded_wdt = math.ceil(width/2)*2
        rec_yuv_path = os.path.join(self.tmp_output_folder, bin_path.replace('.bin', '.yuv'))
        rec_png_path = os.path.join(self.tmp_output_folder, bin_path.replace('.bin', '_tmp.png'))
        output_image_name = os.path.join(self.tmp_output_folder, bin_path.replace('.bin', '.png'))


        # 3. use VTM decoder
        cmd_str= f'{self.decoderApp} -b {bin_path} -o {rec_yuv_path}'
        self.process_cmd(cmd_str, print_out = bPrint)

        # 4. use ffmpeg to convert yuv back to png
        cmd_str= f'ffmpeg -y -f rawvideo -pix_fmt yuv420p10le -s {padded_wdt}x{padded_hgt} -src_range 1 -i {rec_yuv_path} -frames 1  -pix_fmt rgb24 {rec_png_path}'
        self.process_cmd(cmd_str, print_out = bPrint)

        cmd_str= f'ffmpeg -y -i {rec_png_path} -vf "crop={width}:" {output_image_name}'
        self.process_cmd(cmd_str, print_out = bPrint)

        img = Image.open(output_image_name)
        x_hat= transforms.ToTensor()(img).unsqueeze(0)


        return x_hat
        


    def __call_ffmpeg__(self, x, qp):
        bin_path='tmp.bin'
        tensor_size=x.size()
        width= tensor_size[-1]
        height= tensor_size[-2]
        self.__encode_ffmpeg__(x, qp, bin_path, bPrint= False)
        bpp = os.path.getsize(bin_path)
        img = self.__decode_ffmpeg__(bin_path, width, height, bPrint= False)
        return [bpp], img

    def __call__(self, x, qp = None):
        if (qp is None):
            qp= self.qp
        return self.__call_ffmpeg__(x, qp)

    def BGR(self, bgr_image):
        """
        :param bgr_image: numpy BGR image (y,x,3)

        Returns BGR image that has gone through compressai
        """
        rgb_image = bgr_image[:,:,[2,1,0]] # BGR --> RGB
        # rgb_image (y,x,3) to FloatTensor (1,3,y,x):
        # TODO: need to do padding here?
        x = transforms.ToTensor()(rgb_image).unsqueeze(0)
        x = x.to(self.device)
        bpp, x_hat = self(x)
        x_hat = x_hat.squeeze(0).to('cpu')
        rgb_image_hat=np.array(transforms.ToPILImage()(x_hat))
        bgr_image_hat=rgb_image_hat[:,:,[2,1,0]] # RGB --> BGR
        # TODO: need to remove padding / resize here?
        
        if self.save_transformed:
            try:
                os.mkdir("compressai_encoder_decoder")
            except FileExistsError:
                pass
            Image.fromarray(
                bgr_image_hat[:,:,::-1]
                # bgr_image
                    ).save(
                os.path.join("compressai_encoder_decoder","dump_"+str(self.imcount)+".png")
                )
        self.logger.debug("input & output sizes: %s %s. bps = %s", bgr_image.shape, bgr_image_hat.shape, bpp[0])
        # print(">> cc, bpp_sum ", self.cc, self.bpp_sum)
        return bpp[0], bgr_image_hat
