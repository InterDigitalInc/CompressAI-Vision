import logging


class EncoderDecoder:
    """Get input from various frameworks/dataloaders (each implemented in their corresponding methods).

    Transforms images to a format accepted by the encoder/decoder engine (can be a "traditional" encoder, compessai, etc.) & push the image 
    through the encoder/decoder.  Returns bbps & the transformed image(s).

    Subclass for a certain encoder/decoder system.
    
    Create an additional method for a certain famework/dataloder.
    """
    def __init__(self, device = 'cpu'):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.device = device
        self.reset()
        raise(AssertionError("virtual"))


    def reset(self):
        """Reset the internal state of the encoder & decoder, if any
        """
        self.bpp_sum=0 # accumulated bits per pixel sum
        self.cc=0
    

    def getAverageBpps(self):
        """The encoder decoder can internally accumulate bits per pixel sum in order to get the average value
        """
        if self.cc <= 0:
            return 0
        else:
            return self.bpp_sum/self.cc


    def __call__(self, x) -> tuple:
        """Push images(s) through the encoder+decoder, returns bbps and encoded+decoded images

        :param x: a FloatTensor with dimensions (batch, channels, y, x)
        
        Returns (bpps, x_hat), where bpps is a list of bit per pixel values and x_hat is the image that has gone throught the encoder/decoder process
        """
        self.cc += 1
        raise(AssertionError("virtual"))
        return None, None


    def BGR(self, bgr_image):
        """
        :param bgr_image: numpy BGR image (y,x,3)

        Returns BGR image that has gone through transformation (the encoding + decoding process)

        Returns bits_per_pixel, transformed BGR image
        """
        raise(AssertionError("virtual"))



class VoidEncoderDecoder(EncoderDecoder):
    """Does no encoding/decoding .. use for debugging
    """
    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.reset()


    def reset(self):
        """Reset the internal state of the encoder & decoder, if any
        """
        self.bpp_sum=0 # accumulated bits per pixel sum
        self.cc=0
    

    def getAverageBpps(self):
        """The encoder decoder can internally accumulate bits per pixel sum in order to get the average value
        """
        if self.cc <= 0:
            return 0
        else:
            return self.bpp_sum/self.cc


    def __call__(self, x) -> tuple:
        """Push images(s) through the encoder+decoder, returns bbps and encoded+decoded images

        :param x: a FloatTensor with dimensions (batch, channels, y, x)
        
        Returns (bpps, x_hat), where bpps is a list of bit per pixel values and x_hat is the image that has gone throught the encoder/decoder process
        """
        self.cc += 1
        return [0], x


    def BGR(self, bgr_image):
        """
        :param bgr_image: numpy BGR image (y,x,3)

        Returns BGR image that has gone through transformation (the encoding + decoding process)

        Returns bits_per_pixel, transformed BGR image
        """
        return 0, bgr_image




