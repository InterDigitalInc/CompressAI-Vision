import io
import shlex
import subprocess

import numpy as np

from PIL import Image


class FFMpeg:
    """FFMpeg encapsulation

    :param ffmpeg: the ffmpeg command
    :param logger: a logger instance

    TODO: for reading video, define video input, keep ffmpeg process alive & stream into stdin
    """

    def __init__(self, ffmpeg, logger):
        self.ffmpeg = ffmpeg
        self.logger = logger

    def ff_op(self, rgb_image: np.array, op) -> np.array:
        """takes as an input a numpy RGB array (y,x,3)

        Outputs numpy RGB array after certain transformation
        """
        pil_img = Image.fromarray(rgb_image)
        f = io.BytesIO()
        pil_img.save(f, format="png")

        comm = '{ffmpeg} -y -hide_banner -loglevel error -i pipe: -vf "{op}" -f apng pipe:'.format(
            ffmpeg=self.ffmpeg, op=op
        )
        self.logger.debug(comm)
        args = shlex.split(comm)
        p = subprocess.Popen(
            args, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )
        stdout, stderr = p.communicate(f.getvalue())
        if (stdout is None) or (len(stdout) < 5) or p.returncode != 0:
            # print(stderr.decode("utf-8"))
            self.logger.fatal("ffmpeg failed with %s", stderr.decode("utf-8"))
            return None
        f2 = io.BytesIO(stdout)
        pil_img2 = Image.open(f2).convert("RGB")
        return np.array(pil_img2)

    def ff_RGB24ToRAW(self, rgb_image: np.array, form) -> bytes:
        """takes as an input a numpy RGB array (y,x,3)

        ffmpeg -i input.png -f rawvideo -pix_fmt yuv420p -dst_range 1 output.yuv

        produces raw video frame bytes in the given pixel format
        """
        pil_img = Image.fromarray(rgb_image)
        f = io.BytesIO()
        pil_img.save(f, format="png")

        comm = "{ffmpeg} -y -hide_banner -loglevel error -i pipe: -f rawvideo -pix_fmt {form} -dst_range 1 pipe:".format(
            ffmpeg=self.ffmpeg, form=form
        )
        self.logger.debug(comm)
        args = shlex.split(comm)
        p = subprocess.Popen(
            args, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )
        stdout, stderr = p.communicate(f.getvalue())
        if (stdout is None) or (len(stdout) < 5) or p.returncode != 0:  # say
            self.logger.fatal("ffmpeg failed with %s", stderr.decode("utf-8"))
            return None
        f2 = io.BytesIO(stdout)
        return f2.read()

    def ff_RAWToRGB24(self, raw: bytes, form, width=None, height=None) -> bytes:
        """takes as an input a numpy RGB array (y,x,3)

        ffmpeg -y -f rawvideo -pix_fmt yuv420p -s 768x512 -src_range 1 -i test.yuv -frames 1 -pix_fmt rgb24 out.png

        produces RGB image from raw video format
        """
        assert width is not None
        assert height is not None
        assert isinstance(raw, bytes)

        comm = "{ffmpeg} -y -hide_banner -loglevel error -f rawvideo -pix_fmt {form} -s {width}x{height} -src_range 1 -i pipe: -frames 1 -pix_fmt rgb24 -f apng pipe:".format(
            ffmpeg=self.ffmpeg, form=form, width=width, height=height
        )
        self.logger.debug(comm)
        args = shlex.split(comm)
        p = subprocess.Popen(
            args, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )
        stdout, stderr = p.communicate(raw)
        if (stdout is None) or (len(stdout) < 5) or p.returncode != 0:  # say
            self.logger.fatal("ffmpeg failed with %s", stderr.decode("utf-8"))
            return None
        f = io.BytesIO(stdout)
        pil_img = Image.open(f)
        return np.array(pil_img)
