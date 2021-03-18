import cupy as cp
import numpy as np
import sys

from cupy import prof
from scipy import signal
from string import Template

# CuPy: Version 3
# Elementwise kernel for multiple CuPy calls

_signal_kernel = cp.ElementwiseKernel(
    "T signal",
    "T amp, T phase, T real, T imag",
    """
    amp = sqrt(signal.real() * conj(signal));
    phase = arg(signal);
    real = signal.real();
    imag = signal.imag();
    """,
    "_signal_kernel",
    options=("-std=c++11",),
)

def signal(x):
    return _signal_kernel(x)

def cupy_signal(signal):
    amp = cp.asnumpy(cp.sqrt(cp.real(signal * cp.conj(signal))))
    phase = cp.asnumpy(cp.angle(signal))
    real = cp.asnumpy(cp.real(signal))
    imag = cp.asnumpy(cp.imag(signal))
    return amp, phase, real, imag

if __name__ == "__main__":

    #loops = int(sys.argv[1])

    num_samps = (2 ** 16)

    cpu_sig = np.random.rand(num_samps) + 1.0j * np.random.rand(num_samps)
    gpu_sig = cp.array(cpu_sig)

    # Run baseline with signal.seperate
    with prof.time_range("Signal seperate", 0):
        amp, phase, real, imag = cupy_signal(gpu_sig)
        #test = cupy_signal(gpu_sig)
        #print(test)

    # # Run EWK version
    # with prof.time_range("EWK signal", 1):
    #     #amp_EWK, phase_EWK, real_EWK, imag_EWK = signal(gpu_sig)
    #     test2 = signal(gpu_sig)
    #     print(test2)
       
    # Compare results
    #np.testing.assert_allclose(test[0], test2[0] , 1e-3)    
