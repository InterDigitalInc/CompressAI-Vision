import numpy as np
import math, os
import torch
from torchvision import transforms
import torch.nn.functional as F
from PIL import Image
import logging
from .base import EncoderDecoder


class CompressAIEncoderDecoder(EncoderDecoder):
    """EncoderDecoder class for CompressAI
    
    :param net: compressai network, for example:

    ::
    
        net = bmshj2018_factorized(quality=2, pretrained=True).eval().to(device)

    :param device: "cpu" or "cuda"
    :param save_transformed: (debugging) dump transformed images to disk.  default = False
    :param m: images should be multiples of this number.  If not, a padding is applied before passing to compressai.  default = 64
    """
    toFloat=transforms.ConvertImageDtype(torch.float)
    toByte=transforms.ConvertImageDtype(torch.uint8)

    def __init__(self, net, device = 'cpu', save_transformed=False, m:int=64):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.net = net
        self.device = device
        self.save_transformed = save_transformed
        self.m=64
        self.reset()
        self.save_folder="compressai_encoder_decoder"
        if self.save_transformed:
            self.logger.info("Will save images to folder %s", self.save_folder)
            try:
                os.mkdir(self.save_folder)
            except FileExistsError:
                pass
        
    def reset(self):
        """Reset internal image counter
        """
        super().reset()
        self.imcount=0

    def __call__(self, x):
        """Push images(s) through the encoder+decoder, returns bbps and encoded+decoded images

        :param x: a FloatTensor with dimensions (batch, channels, y, x)
        
        Returns (bpps, x_hat), where x_hat is batch of images that have gone through the encoder/decoder process,
        bpps is a list of bits per pixel of each compressed image in that batch

        This method chooses either self.__v0__ or self.__v1__
        """
        # return self.__v0__(x)
        return self.__v1__(x)


    def __v0__(self, x):
        """Push images(s) through the encoder+decoder, returns bbps and encoded+decoded images

        OLD / DEPRECATED VERSION

        :param x: a FloatTensor with dimensions (batch, channels, y, x)
        
        Returns (bpps, x_hat), where x_hat is batch of images that have gone through the encoder/decoder process,
        bpps is a list of bits per second of each compressed image in that batch
        """
        self.logger.debug("feeding tensor to CompressAI: shape: %s, type: %s", x.shape, x.type())

        with torch.no_grad():
            out_net = self.net.forward(x)
        """out_net keys: ['x_hat', 'likelihoods'])

        ::

            out_net["x_hat"] ==> tensor with torch.Size([1, 3, 512, 768]) # batch, channel, y, x .. the transformed image
            out_net["likelihoods"] ==> dict with keys ["y"]
            # .. what if we had fed a larger batch to the network.. .. would "likelihoods" be a list..?
            out_net["likelihoods"]["y"] ==> tensor with torch.Size([1, 192, 32, 48]) # batch, channels, y, x # feature maps


        """
        x_hat = out_net['x_hat'].clamp_(0, 1)
        size = out_net['x_hat'].size()
        num_pixels = size[0] * size[2] * size[3]
        bpps = []
        for likelihood in out_net['likelihoods']["y"]:
            # likelihood / key: "y", value: tensor
            # calculate bpp value
            scaled=torch.log(likelihood).sum() / (-math.log(2) * num_pixels)
            bpp = scaled.item()
            # accumulate bpp_sum for mean value calculation
            self.cc += 1
            self.bpp_sum += bpp
            # print("> cc, bpp_sum ", id(self), self.cc, self.bpp_sum)
            bpps.append(bpp)
        return bpps, x_hat


    def __v1__(self, x):
        """Push images(s) through the encoder+decoder, returns bbps and encoded+decoded images

        :param x: a FloatTensor with dimensions (batch, channels, y, x)
        
        WARNING: we assume that batch=1

        as per: https://github.com/InterDigitalInc/siloai-playground/blob/7b2fe5069abd9489d301647f53e0534f3a7fbfed/jacky/scripts/object_detection_mAP.py#L163

        Returns (bpps, x_hat), where x_hat is batch of images that have gone through the encoder/decoder process,
        bpps is a list of bits per second of each compressed image in that batch
        """
        assert(x.size()[0]==1), "batch dimension must be 1"
        with torch.no_grad():
            # compression
            out_enc = self.net.compress(x)
            # decompression
            out_dec = self.net.decompress(out_enc["strings"], out_enc["shape"])

        # TODO: out_enc["strings"][batch_index?][what?] .. for batch sizes > 1
        bitstream=out_enc["strings"][0][0] # _the_ compressed bitstream
        x_hat = out_dec['x_hat'].clamp(0,1)
        num_pixels=x.shape[2]*x.shape[3]
        # print("num_pixels", num_pixels)
        bpp = 8*len(bitstream)/num_pixels # remember to multiply by eight.. BITS not BYTES
        bpps=[bpp]
        x_hat = out_dec["x_hat"]
        return bpps, x_hat


    def BGR(self, bgr_image: np.array) -> np.array:
        """Return transformed image and bpp for a BGR image

        :param bgr_image: numpy BGR image (y,x,3)

        Returns bits-per-pixel and transformed BGR image that has gone through compressai encoding+decoding.

        Necessary padding for compressai is added and removed on-the-fly
        """
        # TO RGB & TENSOR
        rgb_image = bgr_image[:,:,[2,1,0]] # BGR --> RGB
        # rgb_image (y,x,3) to FloatTensor (1,3,y,x):
        x = transforms.ToTensor()(rgb_image).unsqueeze(0)

        # ADD PADDING
        # padding in order to conform to compressai network
        h, w = x.size(2), x.size(3)
        p = self.m  # maximum 6 strides of 2
        new_h = (h + p - 1) // p * p
        new_w = (w + p - 1) // p * p
        padding_left = (new_w - w) // 2
        padding_right = new_w - w - padding_left
        padding_top = (new_h - h) // 2
        padding_bottom = new_h - h - padding_top
        x_pad = F.pad(
            x,
            (padding_left, padding_right, padding_top, padding_bottom),
            mode="constant",
            value=0,
        )

        # SAVE IMAGE IF
        if self.save_transformed:
            tmp=transforms.ToPILImage()(x_pad.squeeze(0))
            Image.fromarray(
                np.array(tmp) # PIL Image to numpy array 
                    ).save(
                os.path.join(self.save_folder,"dump_pad_"+str(self.imcount)+".png")
                )

        # RUN COMPRESSAI
        x_pad = x_pad.to(self.device)
        bpp, x_hat_pad = self(x_pad)
        x_hat_pad = x_hat_pad.to('cpu')

        # REMOVE PADDING
        # unpad/recover original dimensions
        x_hat = F.pad(
            x_hat_pad, (-padding_left, -padding_right, -padding_top, -padding_bottom)
        )
        assert(x.size() == x_hat.size()), "padding error"

        # TO NUMPY ARRAY & BGR IMAGE
        x_hat= x_hat.squeeze(0)
        rgb_image_hat=np.array(transforms.ToPILImage()(x_hat))
        bgr_image_hat=rgb_image_hat[:,:,[2,1,0]] # RGB --> BGR
        
        # SAVE IMAGE IF
        if self.save_transformed:
            Image.fromarray(
                bgr_image_hat[:,:,::-1]
                # bgr_image
                    ).save(
                os.path.join(self.save_folder,"dump_"+str(self.imcount)+".png")
                )
            self.imcount+=1
        self.logger.debug("input & output sizes: %s %s. bps = %s", bgr_image.shape, bgr_image_hat.shape, bpp[0])
        # print(">> cc, bpp_sum ", self.cc, self.bpp_sum)
        return bpp[0], bgr_image_hat

