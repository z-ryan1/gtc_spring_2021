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
    "T x, int64 n",
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
    x = cp.asarray(x)

    return _gauss_spline_kernel(x, n)

if __name__ == "__main__":

    loops = int(sys.argv[1])

    x = [ 2 ** 16 ]

    in_samps = 2 ** 10
    out_samps = 2 ** 20

    np.random.seed(1234)
    n = np.random.randint(0, 1234)
    x = np.linspace(0.01, 10 * np.pi, in_samps)

    d_x = cp.array(x)
    d_n = cp.array(n)

    # Run baseline with scipy.signal.lombscargle
    with prof.time_range("scipy_gauss_spline", 0):
        cpu_gauss_spline = signal.gauss_spline(x, n)

    # Run Numba version
    with prof.time_range("cupy_gauss_spline", 1):
        gpu_gauss_spline = gauss_spline(d_x, d_n)

    # Copy result to host
    gpu_gauss_spline = cp.asnumpy(gpu_gauss_spline)

    # Compare results
    np.testing.assert_allclose(cpu_gauss_spline, gpu_gauss_spline, 1e-3)    

    # Run multiple passes to get average
    for _ in range(loops):
        with prof.time_range("cupy_gauss_spline_loop", 2):
            gpu_gauss_spline = gauss_spline(d_x, d_n)
