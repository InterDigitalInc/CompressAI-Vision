# The copyright in this software is being made available under the BSD 
# License, included below. This software may be subject to other third party
# and contributor rights, including patent rights, and no such rights are 
# granted under this license.

# Copyright (c) 2022, ISO/IEC
# All rights reserved.

# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:

#  * Redistributions of source code must retain the above copyright notice,
#    this list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#  * Neither the name of the ISO/IEC nor the names of its contributors may 
#    be used to endorse or promote products derived from this software without
#    specific prior written permission.

# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" 
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE 
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS
# BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF
# THE POSSIBILITY OF SUCH DAMAGE.

# References:
# 1. H. Wang et al., "Improvements of the BD-Rate Metrics Using Monotonic Curve-Fitting Methods," 2024 Picture Coding Symposium (PCS), Taichung, Taiwan, 2024, pp. 1-5 (available online at  https://doi.org/10.1109/PCS60826.2024.10566370)


import numpy as np
from scipy.optimize import minimize


def func_cubic_1(x, b):
    return b[0] * np.power(x, 3) + b[1] * np.power(x, 2) + b[2] * np.asarray(x) + b[3]


def fit_cubic(x, y, m_min=0, m_max=100):

    def func_cubic(b, x):
        s = b[0] * np.power(x, 3) + b[1] * np.power(x, 2) + b[2] * np.asarray(x) + b[3]
        return s

    def objective(b):
        return np.sum(np.power(func_cubic(b, x) - y, 2))

    def const_1st_derivative(b):
        return 3 * b[0] * np.power(x, 2) + 2 * b[1] * np.asarray(x) + b[2]

    def const_2nd_derivative(b):
        return -1 * 6 * b[0] * np.asarray(x) - 2 * b[1]

    def const_3rd_minValue(b):
        return (
            b[0] * np.power(x[0], 3)
            + b[1] * np.power(x[0], 2)
            + b[2] * np.asarray(x[0])
            + b[3]
            - m_min
        )

    def const_4th_maxValue(b):
        return m_max - (
            b[0] * np.power(x[-1], 3)
            + b[1] * np.power(x[-1], 2)
            + b[2] * np.asarray(x[-1])
            + b[3]
        )

    cons = (
        dict(type="ineq", fun=const_1st_derivative),
        dict(type="ineq", fun=const_2nd_derivative),
        dict(type="ineq", fun=const_3rd_minValue),
        dict(type="ineq", fun=const_4th_maxValue),
    )
    init = np.array([1.0, 1.0, 1.0, 1.0])
    res = minimize(objective, x0=init, method="SLSQP", constraints=cons)

    return res
