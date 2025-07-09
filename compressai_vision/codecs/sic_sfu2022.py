# Copyright (c) 2022-2024, InterDigital Communications, Inc
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

import logging
import math
import time
import warnings

from pathlib import Path
from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.hub import load_state_dict_from_url

from compressai.ans import BufferedRansEncoder, RansDecoder
from compressai.entropy_models import GaussianConditional
from compressai.layers import (
    MaskedConv2d,
    ResidualBlock,
    ResidualBlockUpsample,
    subpel_conv3x3,
)
from compressai.models.utils import update_registered_buffers
from compressai.models.waseda import Cheng2020Anchor
from compressai_vision.codecs.utils import crop, pad
from compressai_vision.registry import register_multask_codec

from .encdec_utils import (
    read_bytes,
    read_uchars,
    read_uints,
    write_bytes,
    write_uchars,
    write_uints,
)


def filesize(filepath: str) -> int:
    if not Path(filepath).is_file():
        raise ValueError(f'Invalid file "{filepath}".')
    return Path(filepath).stat().st_size


def update_model(model, loaded_state):
    model_dict = model.state_dict()

    pretrained_dict = {k: v for k, v in loaded_state.items() if k in model_dict}
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)


def load_pretrained(model, filename):
    with open(filename, "rb") as f:
        loaded_weights = torch.load(f)
        update_model(model, loaded_weights)


@register_multask_codec("sic_sfu2022")
class SIC_SFU2022:
    def __init__(self, device: str, **kwargs):
        super().__init__()

        self.reset()

        self.logger = logging.getLogger(self.__class__.__name__)
        self.eval_encode = kwargs["eval_encode"]

        self.verbosity = kwargs["verbosity"]
        logging_level = logging.WARN
        if self.verbosity == 1:
            logging_level = logging.INFO
        if self.verbosity >= 2:
            logging_level = logging.DEBUG

        self.logger.setLevel(logging_level)

        # Partsing a string of bottleneck channel
        encoder_config = kwargs["encoder_config"]

        # would there be any better way to do this?
        list_of_lsts = [(key, item) for key, item in encoder_config["strides"].items()]
        list_of_lsts = dict(sorted(list_of_lsts, key=lambda x: x[0]))

        self.vmodels = kwargs["vmodels"]
        self.num_tasks = int(kwargs["num_tasks"])

        # temp
        self.logger.warning(
            "Multi-task compression with SIC SFU2022 is not supported yet"
        )
        raise NotImplementedError

        assert self.num_tasks == 3, "Currently only three tasks model is available"

        if self.num_tasks == 2:
            lst_activations = [nn.Identity()]
        elif self.num_tasks == 3:
            lst_activations = [nn.ReLU(inplace=True), nn.ReLU(inplace=True)]
        else:
            raise NotImplementedError

        strides = []
        for lst in list_of_lsts.values():
            strides.append(tuple(lst))

        self.model = (
            SICHumansMachines(
                SCALABLE_NS=encoder_config["bottleneck_chs"],
                FEATURE_NS=encoder_config["feature_chs"],
                STRIDES_S_F=strides,
                lst_activations=lst_activations,
            )
            .to(device)
            .eval()
        )
        self.device = device

        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.deterministic = True
        torch.set_num_interop_threads(1)  # just to be sure

        # root_url = "https://dspub.blob.core.windows.net/compressai/sic_sfu2022"

        self.target_tlayer = int(kwargs["target_task_layer"])
        assert (
            self.num_tasks == 2 or self.num_tasks == 3
        ), f"SIC_SFU2023 supports only 2 or 3 task layers, but got {self.num_tasks}"
        assert (
            self.target_tlayer < self.num_tasks
        ), f"target task layer must be lower than the number of tasks, \
            but got {self.target_tlayer} < {self.num_tasks}"

        self.trg_vmodel = self.vmodels[self.target_tlayer]

        self.ftensor_alignment_size = 0
        if self.target_tlayer < (self.num_tasks - 1):
            self.ftensor_alignment_size = self.trg_vmodel.size_divisibility

        weights_url = {None: None}

        self.padding_size = 64

        self.qidx = encoder_config["qidx"]
        model_weight_url = weights_url[self.num_tasks][self.qidx]
        weight = load_state_dict_from_url(
            model_weight_url, progress=True, check_hash=True, map_location=device
        )
        self.update_model(self.model, weight)

    @staticmethod
    def get_padded_input_size(fSize, p):
        h, w = fSize
        H = ((h + p - 1) // p) * p
        W = ((w + p - 1) // p) * p

        return (H, W)

    @property
    def eval_encode_type(self):
        return self.eval_encode

    @property
    def qp_value(self):
        return self.qidx

    @staticmethod
    def update_model(model, loaded_state):
        model_dict = model.state_dict()

        pretrained_dict = {k: v for k, v in loaded_state.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
        model.update()

    @staticmethod
    def load_pretrained(model, filename):
        with open(filename, "rb") as f:
            loaded_weights = torch.load(f)
            update_model(model, loaded_weights)

    def reset(self):
        self.target_tlayer = -1
        self.num_tasks = -1

    def encode(
        self,
        x: Dict,
        codec_output_dir,
        bitstream_name,
        file_prefix: str = "",
    ):
        if file_prefix == "":
            file_prefix = f"{codec_output_dir}/{bitstream_name}"
        else:
            file_prefix = f"{codec_output_dir}/{bitstream_name}-{file_prefix}"

        # logpath = Path(f"{file_prefix}_enc.log")

        if self.trg_vmodel is None:
            img = x["image"].to(self.device)
            assert img.dim() == 3
            feature_only = False
        else:
            assert self.target_tlayer < (self.num_tasks - 1)

            img = (
                self.trg_vmodel.input_resize(
                    [x["image"].to(self.device).float()]
                ).tensor
            )[0]

            img = img[[2, 1, 0], :, :] / 255.0
            feature_only = True

        input_fh, input_fw = img.shape[1:]

        start = time.perf_counter()
        with torch.no_grad():
            img = pad(img.unsqueeze(0), self.padding_size, bottom_right=feature_only)
            out = self.model.compress(
                img, self.target_tlayer, feature_only=feature_only
            )

        all_pathes = []
        all_bytes = []
        accum_bytes = 0
        for lid in range(0, self.target_tlayer + 1):
            bitstream_path = f"{file_prefix}_l{lid}.bin"
            all_pathes.append(bitstream_path)
            with Path(bitstream_path).open("wb") as f:
                if lid == 0:
                    # write original image size
                    write_uints(f, (x["height"], x["width"]))
                    # write input image size
                    write_uints(f, (input_fh, input_fw))
                    # write the total number of tasks
                    write_uchars(f, (self.num_tasks,))
                    # write active layer id
                    write_uchars(f, (self.target_tlayer,))

                    # write a shape info
                    write_uints(f, (out["shape"][0], out["shape"][1]))
                    # a single hyperprior for all
                    write_uints(f, (len(out["strings"][1][0]),))
                    write_bytes(f, out["strings"][1][0])

                write_uints(f, (len(out["strings"][0][lid][0]),))
                write_bytes(f, out["strings"][0][lid][0])

            size = filesize(bitstream_path)
            cbytes = float(size)
            accum_bytes += cbytes
            all_bytes.append(cbytes)

        enc_time = time.perf_counter() - start

        self.logger.debug(f"enc_time:{enc_time}")

        return {
            "bytes": [accum_bytes],
            "bitstream": all_pathes,
        }

    def decode(
        self,
        bitstream_path: Path = None,
        codec_output_dir: str = "",
        file_prefix: str = "",
    ) -> bool:
        self.reset()

        assert isinstance(bitstream_path, list)

        start = time.perf_counter()

        main_strings = []
        for e, bp in enumerate(bitstream_path):
            b_path = Path(bp)
            assert b_path.is_file()

            with b_path.open("rb") as f:
                if e == 0:
                    # output_file_prefix = b_path.stem

                    # read original image size
                    org_fh, org_fw = read_uints(f, 2)

                    # read input image size
                    input_fh, input_fw = read_uints(f, 2)

                    # read the total number of tasks
                    self.num_tasks = read_uchars(f, 1)[0]

                    # read active layer id
                    self.target_tlayer = read_uchars(f, 1)[0]

                    # read a shape info
                    shape = read_uints(f, 2)

                    # a single hyperprior for all
                    nbytes = read_uints(f, 1)[0]
                    hyperprior_string = [read_bytes(f, nbytes)]

                nbytes = read_uints(f, 1)[0]
                main_strings.append([read_bytes(f, nbytes)])

        with torch.no_grad():
            out = self.model.decompress([main_strings, hyperprior_string], shape)

        dec_time = time.perf_counter() - start
        self.logger.debug(f"dec_time:{dec_time}")

        if self.target_tlayer == (self.num_tasks - 1):  # Reconstruction for image
            assert f"l{self.target_tlayer}" in out
            out["l2"] = crop(out["l2"], (org_fh, org_fw))
        else:  # estimated features
            est_features = out[f"l{self.target_tlayer}"]

            est_fH, est_fW = est_features.shape[2:]
            pad_iH, pad_iW = self.get_padded_input_size(
                (input_fh, input_fw), self.padding_size
            )

            assert (pad_iH / est_fH) == (pad_iW / est_fW)
            scale = int(pad_iH / est_fH)

            pad_fH, pad_fW = self.get_padded_input_size(
                (input_fh, input_fw), self.ftensor_alignment_size
            )

            out[f"l{self.target_tlayer}"] = crop(
                out[f"l{self.target_tlayer}"],
                (pad_fH // scale, pad_fW // scale),
                bottom_right=True,
            )

        output = {
            "data": out,
            "org_input_size": {"height": org_fh, "width": org_fw},
            "input_size": [(input_fh, input_fw)],
        }
        return output


# From Balle's tensorflow compression examples
SCALES_MIN = 0.11
SCALES_MAX = 256
SCALES_LEVELS = 64


def get_scale_table(min=SCALES_MIN, max=SCALES_MAX, levels=SCALES_LEVELS):  # pylint: disable=W0622
    return torch.exp(torch.linspace(math.log(min), math.log(max), levels))


class SICHumansMachines(Cheng2020Anchor):
    """End-to-end image codec for humans and machines from `Scalable Image
    Coding for Humans and Machines"
    <https://ieeexplore.ieee.org/document/9741390>`_,
    by Hyomin Choi and Ivan V. Bajić.

    Uses residual blocks with small convolutions (3x3 and 1x1), and sub-pixel
    convolutions for up-sampling.

    Args:
        SCALABLE_NS (list of ints): A list of number of bottleneck channels for each layer.
        FEATURE_NS (list of ints):
        STRIDES_S_F (list of tuples):
    """

    def __init__(
        self,
        SCALABLE_NS: list,
        FEATURE_NS: list,
        STRIDES_S_F: list,
        lst_activations: list,
        **kwargs,
    ):
        assert is_sequence(SCALABLE_NS)
        assert is_sequence(FEATURE_NS)
        assert is_sequence(STRIDES_S_F)

        assert len(SCALABLE_NS) >= 1
        assert len(FEATURE_NS) == (len(SCALABLE_NS) - 1)
        assert len(FEATURE_NS) == len(STRIDES_S_F)

        for strides in STRIDES_S_F:
            assert is_sequence(strides)

        self.TB_N = sum(SCALABLE_NS)
        self.SCALABLE_NS = SCALABLE_NS
        self.NUM_LAYERS = len(SCALABLE_NS)

        kwargs.pop("N", None)
        super().__init__(N=self.TB_N, **kwargs)

        class LatentSpaceTransform(nn.Module):
            def __init__(
                self,
                bottleneck_n: int,
                feature_n: int,
                strides: tuple,
                activation=nn.Identity(),
            ):
                super().__init__()

                self.lst = nn.Sequential(
                    ResidualBlock(bottleneck_n, bottleneck_n),
                    ResidualBlockUpsample(bottleneck_n, bottleneck_n, strides[0]),
                    ResidualBlock(bottleneck_n, bottleneck_n),
                    ResidualBlockUpsample(bottleneck_n, bottleneck_n, strides[1]),
                    ResidualBlock(bottleneck_n, bottleneck_n),
                    ResidualBlockUpsample(bottleneck_n, bottleneck_n, strides[2]),
                    ResidualBlock(bottleneck_n, bottleneck_n),
                    subpel_conv3x3(bottleneck_n, feature_n, strides[3]),
                    activation,
                )

            def forward(self, x):
                return self.lst(x)

        class EntropyModules(nn.Module):
            def __init__(self, bottleneck_n: int):
                super(EntropyModules, self).__init__()
                context_prediction = MaskedConv2d(
                    bottleneck_n, 2 * bottleneck_n, kernel_size=5, padding=2, stride=1
                )

                entropy_params = nn.Sequential(
                    nn.Conv2d(bottleneck_n * 12 // 3, bottleneck_n * 10 // 3, 1),
                    nn.LeakyReLU(inplace=True),
                    nn.Conv2d(bottleneck_n * 10 // 3, bottleneck_n * 8 // 3, 1),
                    nn.LeakyReLU(inplace=True),
                    nn.Conv2d(
                        bottleneck_n * 8 // 3, bottleneck_n * 6 // 3, 1
                    ),  # 3 * K * N (Mixed Gaussian), K = 3
                )

                gaussian_conditional = GaussianConditional(None)
                self.entp_modules = nn.ModuleDict(
                    {
                        "context_prediction": context_prediction,
                        "entropy_parameters": entropy_params,
                        "gaussian_conditional": gaussian_conditional,
                    }
                )

        self.entp = nn.ModuleList(
            [EntropyModules(SCALABLE_NS[e]) for e in range(self.NUM_LAYERS)]
        )

        assert len(lst_activations) == (self.NUM_LAYERS - 1)

        self.lsts = nn.ModuleList(
            [
                LatentSpaceTransform(
                    sum(SCALABLE_NS[: (e + 1)]),
                    FEATURE_NS[e],
                    STRIDES_S_F[e],
                    lst_activations[e],
                )
                for e in range(self.NUM_LAYERS - 1)
            ]
        )

        self.entropy_parameters = None
        self.gaussian_conditional = None
        self.context_prediction = None

    def getNumLayers(self):
        return self.NUM_LAYERS

    def forward(self, x):
        y = self.g_a(x)

        per_layer_y = []
        start_ch_y, end_ch_y = 0, 0
        for e in range(self.NUM_LAYERS):
            end_ch_y = start_ch_y + self.SCALABLE_NS[e]
            per_layer_y.append(y[:, start_ch_y:end_ch_y, :, :])
            start_ch_y = start_ch_y + self.SCALABLE_NS[e]

        z = self.h_a(y)
        z_hat, z_likelihoods = self.entropy_bottleneck(z)
        params = self.h_s(z_hat)

        y_hats = None
        output_hats = {}
        output_likelihoods = {"z": z_likelihoods}

        start_ch_param, end_ch_param = 0, 0
        for e in range(self.NUM_LAYERS):
            end_ch_param = start_ch_param + (self.SCALABLE_NS[e] * 2)

            per_layer_params = params[:, start_ch_param:end_ch_param, :, :]
            per_layer_y_hat = (
                self.entp[e]
                .entp_modules["gaussian_conditional"]
                ._quantize(per_layer_y[e], "noise" if self.training else "dequantize")
            )
            per_layer_ctx_params = self.entp[e].entp_modules["context_prediction"](
                per_layer_y_hat
            )
            per_layer_gaussian_params = self.entp[e].entp_modules["entropy_parameters"](
                torch.cat((per_layer_params, per_layer_ctx_params), dim=1)
            )
            scales, means = per_layer_gaussian_params.chunk(2, 1)
            _, per_layer_y_likelihoods = self.entp[e].entp_modules[
                "gaussian_conditional"
            ](per_layer_y[e], scales, means=means)

            output_likelihoods.update({f"l{e}": per_layer_y_likelihoods})

            y_hats = (
                per_layer_y_hat
                if y_hats is None
                else torch.cat((y_hats, per_layer_y_hat), dim=1)
            )
            if e < (self.NUM_LAYERS - 1):
                output_hat = {f"l{e}": self.lsts[e](y_hats)}
            else:
                assert e == (self.NUM_LAYERS - 1)
                output_hat = {f"l{e}": self.g_s(y_hats)}

            output_hats.update(output_hat)

            start_ch_param = start_ch_param + (self.SCALABLE_NS[e] * 2)

        return {"out_hats": output_hats, "likelihoods": output_likelihoods}

    @classmethod
    def from_state_dict(cls, state_dict):
        """Return a new model instance from `state_dict`."""
        N = state_dict["g_a.0.weight"].size(0)
        M = state_dict["g_a.6.weight"].size(0)
        net = cls(N, M)
        net.load_state_dict(state_dict)
        return net

    def compress(self, x, target_layer=0, feature_only=False):
        if next(self.parameters()).device != torch.device("cpu"):
            warnings.warn(
                "Inference on GPU is not recommended for the autoregressive "
                "models (the entropy coder is run sequentially on CPU)."
            )

        assert (
            target_layer < self.NUM_LAYERS
        ), f"Got the target layer {target_layer}, but should be less than {self.NUM_LAYERS}"

        y = self.g_a(x)
        z = self.h_a(y)

        z_strings = self.entropy_bottleneck.compress(z)
        z_hat = self.entropy_bottleneck.decompress(z_strings, z.size()[-2:])
        params = self.h_s(z_hat)

        per_layer_y = []
        per_layer_param = []
        start_ch_y, end_ch_y = 0, 0
        start_ch_param, end_ch_param = 0, 0
        for e in range(self.NUM_LAYERS):
            end_ch_y = start_ch_y + self.SCALABLE_NS[e]
            end_ch_param = start_ch_param + (self.SCALABLE_NS[e] * 2)

            per_layer_y.append(y[:, start_ch_y:end_ch_y, :, :])
            per_layer_param.append(params[:, start_ch_param:end_ch_param, :, :])

            start_ch_y = start_ch_y + self.SCALABLE_NS[e]
            start_ch_param = start_ch_param + (self.SCALABLE_NS[e] * 2)

        s = 4  # scaling factor between z and y
        kernel_size = 5  # context prediction kernel size
        padding = (kernel_size - 1) // 2

        y_height = z_hat.size(2) * s
        y_width = z_hat.size(3) * s

        output_strings = []
        # active_num_layers = (
        #    (self.NUM_LAYERS - 1) if feature_only is True else self.NUM_LAYERS
        # )
        active_num_layers = target_layer + 1

        for e in range(active_num_layers):
            per_layer_y_hat = F.pad(
                per_layer_y[e], (padding, padding, padding, padding), "constant", 0
            )

            per_layer_y_strings = []
            for i in range(per_layer_y[e].size(0)):
                string, _ = self._compress_ar(
                    e,
                    per_layer_y_hat[i : i + 1],
                    per_layer_param[e][i : i + 1],
                    y_height,
                    y_width,
                    kernel_size,
                    padding,
                )
                per_layer_y_strings.append(string)

            output_strings.append(per_layer_y_strings)

        return {"strings": [output_strings, z_strings], "shape": z.size()[-2:]}

    def _compress_ar(self, ldx, y_hat, params, height, width, kernel_size, padding):
        gaussian_conditional = self.entp[ldx].entp_modules["gaussian_conditional"]
        context_prediction = self.entp[ldx].entp_modules["context_prediction"]
        entropy_params = self.entp[ldx].entp_modules["entropy_parameters"]

        cdf = gaussian_conditional.quantized_cdf.tolist()
        cdf_lengths = gaussian_conditional.cdf_length.tolist()
        offsets = gaussian_conditional.offset.tolist()

        encoder = BufferedRansEncoder()
        symbols_list = []
        indexes_list = []

        # Warning, this is slow...
        # TODO: profile the calls to the bindings...
        masked_weight = context_prediction.weight * context_prediction.mask

        for h in range(height):
            for w in range(width):
                y_crop = y_hat[:, :, h : h + kernel_size, w : w + kernel_size]
                ctx_p = F.conv2d(
                    y_crop * context_prediction.mask[0:1, :, :, :],
                    masked_weight,
                    bias=context_prediction.bias,
                )

                # 1x1 conv for the entropy parameters prediction network, so
                # we only keep the elements in the "center"
                p = params[:, :, h : h + 1, w : w + 1]
                gaussian_params = entropy_params(torch.cat((p, ctx_p), dim=1))
                gaussian_params = gaussian_params.squeeze(3).squeeze(2)
                scales_hat, means_hat = gaussian_params.chunk(2, 1)

                indexes = gaussian_conditional.build_indexes(scales_hat)

                y_crop = y_crop[:, :, padding, padding]
                y_q = gaussian_conditional.quantize(y_crop, "symbols", means_hat)
                y_hat[:, :, h + padding, w + padding] = y_q + means_hat

                symbols_list.extend(y_q.squeeze().tolist())
                indexes_list.extend(indexes.squeeze().tolist())

        encoder.encode_with_indexes(
            symbols_list, indexes_list, cdf, cdf_lengths, offsets
        )

        string = encoder.flush()
        return string, y_hat

    def decompress(self, strings, shape):
        assert isinstance(strings, list) and len(strings) == 2
        assert type(strings[0]) is list

        if next(self.parameters()).device != torch.device("cpu"):
            warnings.warn(
                "Inference on GPU is not recommended for the autoregressive "
                "models (the entropy coder is run sequentially on CPU)."
            )

        # FIXME: we don't respect the default entropy coder and directly call the
        # range ANS decoder

        z_hat = self.entropy_bottleneck.decompress(strings[1], shape)
        params = self.h_s(z_hat)

        s = 4  # scaling factor between z and y
        kernel_size = 5  # context prediction kernel size
        padding = (kernel_size - 1) // 2

        y_height = z_hat.size(2) * s
        y_width = z_hat.size(3) * s

        output_hats = {}
        y_hat = None
        start_ch_param, end_ch_param = 0, 0
        for e, main_string in enumerate(strings[0]):
            # initialize y_hat to zeros, and pad it so we can directly work with
            # sub-tensors of size (N, C, kernel size, kernel_size)

            end_ch_param = start_ch_param + (self.SCALABLE_NS[e] * 2)

            per_layer_y_hat = torch.zeros(
                (
                    z_hat.size(0),
                    self.SCALABLE_NS[e],
                    y_height + 2 * padding,
                    y_width + 2 * padding,
                ),
                device=z_hat.device,
            )
            per_layer_param = params[:, start_ch_param:end_ch_param, :, :]

            start_ch_param = start_ch_param + (self.SCALABLE_NS[e] * 2)

            for i, y_string in enumerate(main_string):
                per_layer_y_hat = self._decompress_ar(
                    e,
                    y_string,
                    per_layer_y_hat[i : i + 1],
                    per_layer_param[i : i + 1],
                    y_height,
                    y_width,
                    kernel_size,
                    padding,
                )

            y_hat = (
                per_layer_y_hat
                if y_hat is None
                else torch.cat((y_hat, per_layer_y_hat), dim=1)
            )

            if e < (self.NUM_LAYERS - 1):
                output_hat = {
                    f"l{e}": self.lsts[e](
                        F.pad(y_hat, (-padding, -padding, -padding, -padding))
                    )
                }
            else:
                assert e == (self.NUM_LAYERS - 1)
                x_hat = self.g_s(
                    F.pad(y_hat, (-padding, -padding, -padding, -padding))
                ).clamp_(0, 1)

                output_hat = {f"l{e}": x_hat}

            output_hats.update(output_hat)

        return output_hats

    def _decompress_ar(
        self, ldx, y_string, y_hat, params, height, width, kernel_size, padding
    ):
        gaussian_conditional = self.entp[ldx].entp_modules["gaussian_conditional"]
        context_prediction = self.entp[ldx].entp_modules["context_prediction"]
        entropy_params = self.entp[ldx].entp_modules["entropy_parameters"]

        cdf = gaussian_conditional.quantized_cdf.tolist()
        cdf_lengths = gaussian_conditional.cdf_length.tolist()
        offsets = gaussian_conditional.offset.tolist()

        decoder = RansDecoder()
        decoder.set_stream(y_string)

        # Warning: this is slow due to the auto-regressive nature of the
        # decoding... See more recent publication where they use an
        # auto-regressive module on chunks of channels for faster decoding...

        masked_weight = context_prediction.weight * context_prediction.mask
        for h in range(height):
            for w in range(width):
                # only perform the 5x5 convolution on a cropped tensor
                # centered in (h, w)
                y_crop = y_hat[:, :, h : h + kernel_size, w : w + kernel_size]
                ctx_p = F.conv2d(
                    y_crop,
                    masked_weight,
                    bias=context_prediction.bias,
                )

                # 1x1 conv for the entropy parameters prediction network, so
                # we only keep the elements in the "center"
                p = params[:, :, h : h + 1, w : w + 1]
                gaussian_params = entropy_params(torch.cat((p, ctx_p), dim=1))
                scales_hat, means_hat = gaussian_params.chunk(2, 1)

                indexes = gaussian_conditional.build_indexes(scales_hat)
                rv = decoder.decode_stream(
                    indexes.squeeze().tolist(), cdf, cdf_lengths, offsets
                )
                rv = torch.Tensor(rv).reshape(1, -1, 1, 1)
                rv = gaussian_conditional.dequantize(rv, means_hat)

                hp = h + padding
                wp = w + padding
                y_hat[:, :, hp : hp + 1, wp : wp + 1] = rv

        return y_hat

    def update(self, scale_table=None, force=False):
        if scale_table is None:
            scale_table = get_scale_table()

        for e, modules in enumerate(self.entp):
            modules.entp_modules["gaussian_conditional"].update_scale_table(
                scale_table, force=force
            )
        self.entropy_bottleneck.update(force)
        # super().update(force=force)

    def load_state_dict(self, state_dict):
        # Dynamically update the entropy bottleneck buffers related to the CDFs

        update_registered_buffers(
            self.entropy_bottleneck,
            "entropy_bottleneck",
            ["_quantized_cdf", "_offset", "_cdf_length"],
            state_dict,
        )

        for e, entp in enumerate(self.entp):
            gaussian_conditional = entp.entp_modules["gaussian_conditional"]

            update_registered_buffers(
                gaussian_conditional,
                f"entp.{e}.entp_modules.gaussian_conditional",
                ["_quantized_cdf", "_offset", "_cdf_length", "scale_table"],
                state_dict,
            )
            # gaussian_conditional.load_state_dict(state_dict)

        # super().load_state_dict(state_dict)
        nn.Module.load_state_dict(self, state_dict)

        return self


def is_sequence(x):
    return (
        not hasattr(x, "strip") and hasattr(x, "__getitem__") or hasattr(x, "__iter__")
    )
