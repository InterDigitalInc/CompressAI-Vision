# This file is covered by the license agreement found in the file “license.txt” in the root of this project.
# collect results in a format that can be paste to the reporting template


# source codes of MPEG WG4 VCM proposal m63692
# Ding Ding, et al., "[VCM] Improvements of the BD-rate model using monotonic
# curve-fitting method," ISO/IEC JTC 1/SC 29/WG 4, Doc. m63692，Geneva, CH – July 2023.
# contact: ddding@tencent.com

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
