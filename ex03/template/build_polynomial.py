# -*- coding: utf-8 -*-
"""implement a polynomial basis function."""

import numpy as np


def build_poly(x, degree):
    """polynomial basis functions for input data x, for j=0 up to j=degree."""
    res = np.zeros(shape=(x.shape[0], degree+1))
    for i in range(x.shape[0]):
        for e in range(degree+1):
            res[i, e] = x[i]**e
    return res
