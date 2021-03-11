import numpy as np
import sys
import cupy as cp
from cupy import prof
from scipy import signal

# Python: Version 1
# Naive serial implementation of Python

if __name__ == "__main__":
    
    x = [ 2 ** 16 ]
    in_samps = 2 ** 10
    out_samps = 2 ** 20

    np.random.seed(1234)
    n = np.random.randint(0, 1234)
    x = np.linspace(0.01, 10 * np.pi, in_samps)

    # # Use float32 if b32 passed
    # if dtype == 'float32':
    #     x = x.astype(np.float32)
    #     n = n.astype(np.float32)

    with prof.time_range("scipy_gauss_spline", 0):
        cpu_gauss_spline = signal.gauss_spline(x, n)

    # Run baseline with scipy.signal.gauss_spline
    for _ in range(100):
        with prof.time_range("scipy_gauss_spline_loop", 0):
            cpu_gauss_spline = signal.gauss_spline(x, n)