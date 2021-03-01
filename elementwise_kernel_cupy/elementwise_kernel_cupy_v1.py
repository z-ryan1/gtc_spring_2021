import cupy as cp
import numpy as np
import sys

from cupy import prof
from math import sin, cos, atan2
from scipy import signal
from string import Template

# CuPy: Version 2
# Elementwise kernel implementation of CuPy

_gauss_spline_kernel = cp.ElementwiseKernel(
    "T x, int32 n",
    "T output",
    """
    output = 1 / sqrt( 2.0 * M_PI * signsq ) * exp( -( x * x ) * r_signsq );
    """,
    "_gauss_spline_kernel",
    options=("-std=c++11",),
    loop_prep="const double signsq { ( n + 1 ) / 12.0 }; \
               const double r_signsq { 0.5 / signsq };",
)


def gauss_spline(x, n):
    """Gaussian approximation to B-spline basis function of order n.
    Parameters
    ----------
    n : int
        The order of the spline. Must be nonnegative, i.e. n >= 0
    References
    ----------
    .. [1] Bouma H., Vilanova A., Bescos J.O., ter Haar Romeny B.M., Gerritsen
       F.A. (2007) Fast and Accurate Gaussian Derivatives Based on B-Splines.
       In: Sgallari F., Murli A., Paragios N. (eds) Scale Space and Variational
       Methods in Computer Vision. SSVM 2007. Lecture Notes in Computer
       Science, vol 4485. Springer, Berlin, Heidelberg
    """
    x = cp.asarray(x)

    return _gauss_spline_kernel(x, n)
