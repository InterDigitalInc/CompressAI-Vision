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

import copy
import logging

import numpy as np
import torch

from torch.nn import functional as F


class IntConv2d(torch.nn.Conv2d):
    def __init__(self, *args, **kwargs) -> None:
        _nkwargs = copy.deepcopy(kwargs)

        if _nkwargs.get("training") is not None:
            del _nkwargs["training"]

        if _nkwargs.get("transposed") is not None:
            del _nkwargs["transposed"]

        if _nkwargs.get("output_padding") is not None:
            del _nkwargs["output_padding"]

        for name in kwargs.keys():
            if name.startswith("_"):
                del _nkwargs[name]

        super().__init__(*args, **_nkwargs)
        self.initified_weight_mode = False

    def quantize_weights(self):
        self.initified_weight_mode = True

        if self.bias is None:
            self.float_bias = torch.zeros(self.out_channels, device=self.weight.device)
        else:
            self.float_bias = self.bias.detach().clone()

        if self.weight.dtype == torch.float32:
            _precision = 2 ** (23 + 1)
        elif self.weight.dtype == torch.float64:
            _precision = 2 ** (52 + 1)
        else:
            logging.warning(
                f"Unsupported dtype {self.weight.dtype}. Behaviour may lead unexpected results."
            )
            _precision = 2 ** (23 + 1)

        ###### ADOPT VCMRS IMPLEMENTATION ######
        # sf const
        sf_const = 48

        N = np.prod(self.weight.shape[1:])
        self.N = N
        self.factor = np.sqrt(_precision)
        # self.sf = 1/6 #precision bits allocation factor
        self.sf = np.sqrt(sf_const / N)

        # perform the calculate ion CPU to stabalize the calculation
        self.w_sum = self.weight.cpu().abs().sum(axis=[1, 2, 3]).to(self.weight.device)
        self.w_sum[self.w_sum == 0] = 1  # prevent divide by 0

        self.fw = (self.factor / self.sf - np.sqrt(N / 12) * 5) / self.w_sum

        # intify weights
        self.weight.requires_grad = False  # Just make sure
        self.weight.copy_(
            torch.round(self.weight.detach().clone() * self.fw.view(-1, 1, 1, 1))
        )

        # set bias to 0
        if self.bias is not None:
            self.bias.requires_grad = False  # Just make sure
            self.bias.zero_()

        ###### END OF THE REFERENCE IMPELEMENTATION OF THE INT CONVS IN VCMRS ######

    def integer_conv2d(self, x: torch.Tensor):
        _dtype = x.dtype
        _cudnn_enabled = torch.backends.cudnn.enabled
        torch.backends.cudnn.enabled = False

        ######  ADOPT VCMRS IMPLEMENTATION  ######
        # Calculate factor
        fx = 1

        x_abs = x.abs()
        x_max = x_abs.max()
        if x_max > 0:
            fx = (self.factor * self.sf - 0.5) / x_max

        # intify x
        out_x = torch.round(fx * x)

        out_x = F.conv2d(
            out_x,
            self.weight,
            self.bias,
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
        )

        # x should be all integers
        out_x = out_x / (fx * self.fw.to(out_x.device).view(-1, 1, 1)).float()

        # apply bias in float format
        out_x = (
            (out_x.permute(0, 2, 3, 1) + self.float_bias.to(out_x.device))
            .permute(0, 3, 1, 2)
            .contiguous()
        )
        ###### END OF THE REFERENCE IMPELEMENTATION OF THE INT CONVS IN VCMRS ######
        torch.backends.cudnn.enabled = _cudnn_enabled

        return out_x.to(_dtype)

    def conv2d(self, x: torch.Tensor):
        return F.conv2d(
            x,
            self.weight,
            self.bias,
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
        )


class IntTransposedConv2d(torch.nn.ConvTranspose2d):
    def __init__(self, *args, **kwargs) -> None:
        _nkwargs = copy.deepcopy(kwargs)

        if _nkwargs.get("training") is not None:
            del _nkwargs["training"]

        if _nkwargs.get("transposed") is not None:
            del _nkwargs["transposed"]

        for name in kwargs.keys():
            if name.startswith("_"):
                del _nkwargs[name]

        super().__init__(*args, **_nkwargs)
        self.initified_weight_mode = False

    # prepare quantized weights
    def quantize_weights(self):
        self.initified_weight_mode = True

        if self.bias is None:
            self.float_bias = torch.zeros(self.out_channels, device=self.weight.device)
        else:
            self.float_bias = self.bias.detach().clone()

        if self.weight.dtype == torch.float32:
            _precision = 2 ** (23 + 1)
        elif self.weight.dtype == torch.float64:
            _precision = 2 ** (52 + 1)
        else:
            logging.warning(
                f"Unsupported dtype {self.weight.dtype}. Behaviour may lead unexpected results."
            )
            _precision = 2 ** (23 + 1)

        ######  ADOPT VCMRS IMPLEMENTATION  ######
        # sf const
        sf_const = 48

        N = np.prod(self.weight.shape) / self.weight.shape[1]  # (in, out, kH, kW)
        self.N = N
        self.factor = np.sqrt(_precision)
        # self.sf = 1/6 #precision bits allocation factor
        self.sf = np.sqrt(sf_const / N)

        # perform the calculate ion CPU to stabalize the calculation
        self.w_sum = self.weight.cpu().abs().sum(axis=[0, 2, 3]).to(self.weight.device)
        self.w_sum[self.w_sum == 0] = 1  # prevent divide by 0

        self.fw = (self.factor / self.sf - np.sqrt(N / 12) * 5) / self.w_sum

        # intify weights
        self.weight.requires_grad = False  # Just make sure
        self.weight.copy_(
            torch.round(self.weight.detach().clone() * self.fw.view(1, -1, 1, 1))
        )

        # set bias to 0
        if self.bias is not None:
            self.bias.requires_grad = False  # Just make sure
            self.bias.zero_()

        ###### END OF THE REFERENCE IMPELEMENTATION OF THE INT CONVS IN VCMRS ######

    def integer_transposeconv2d(self, x: torch.Tensor):
        _dtype = x.dtype
        _cudnn_enabled = torch.backends.cudnn.enabled
        torch.backends.cudnn.enabled = False

        ######  ADOPT VCMRS IMPLEMENTATION  ######
        # Calculate factor
        fx = 1

        x_abs = x.abs()
        x_max = x_abs.max()
        if x_max > 0:
            fx = (self.factor * self.sf - 0.5) / x_max

        # intify x
        out_x = torch.round(fx * x)
        out_x = super().forward(out_x)

        # x should be all integers
        out_x = out_x / (fx * self.fw.to(out_x.device).view(-1, 1, 1))
        out_x = out_x.float()

        # apply bias in float format
        out_x = (
            (out_x.permute(0, 2, 3, 1) + self.float_bias.to(out_x.device))
            .permute(0, 3, 1, 2)
            .contiguous()
        )

        ###### END OF THE REFERENCE IMPELEMENTATION OF THE INT CONVS IN VCMRS ######
        torch.backends.cudnn.enabled = _cudnn_enabled

        return out_x.to(_dtype)

    def transposedconv2d(self, x: torch.Tensor):
        return super().forward(x)
