# Copyright (c) 2019-2020, NVIDIA CORPORATION.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import cupy as cp
import numpy as np
import sys

from cupy import prof
from scipy import signal
from math import pi 
from string import Template

# CuPy: Version 1
# Naive implementation of CuPy


_cupy_gauss_spline_src = Template(
    """
extern "C" {
    __global__ void _cupy_gauss_spline(
            const int x_shape,
            const float * __restrict__ x,
            const int n,
            float * __restrict__ res
            ) {
            
        const int tx {
            static_cast<int>( blockIdx.x * blockDim.x + threadIdx.x ) };
        const int stride { static_cast<int>( blockDim.x * gridDim.x ) };

        const float  PI = 3.14159265358979f;
        for ( int tid = tx; tid < x_shape; tid += stride) {
            float x_val { x[tid] };
            int signsq {};
            float val {};
            float val2 {};
            float val3 {};
            float var {};

            signsq = (n + 1) / 12.0;
            val = 1 / sqrt( 2 * signsq * PI );
            var = 2 / 2 / signsq;
            val3 = exp( pow( -x_val, var ) ) ;
            res[tid] = (
                val * val3
            );
        }
    }
}
"""
)

def _gauss_spline(x, n, res):

    device_id = cp.cuda.Device()
    numSM = device_id.attributes["MultiProcessorCount"]
    threadsperblock = (128, )
    blockspergrid = (numSM * 20,)
    
    src = _cupy_gauss_spline_src.substitute(datatype=float)
    module = cp.RawModule(code=src, options=("-std=c++11",))
    kernel = module.get_function("_cupy_gauss_spline")

    kernel_args = (
            x.shape[0],
            x,
            n,
            res,
        )

    kernel(blockspergrid, threadsperblock, kernel_args)
    cp.cuda.runtime.deviceSynchronize()


def gauss_spline(
    x,
    n,
):
    res = cp.empty(x.shape[0], dtype=x.dtype)
    _gauss_spline(x, n, res)

    return res


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
        print(gpu_gauss_spline)

    # Copy result to host
    gpu_gauss_spline = cp.asnumpy(gpu_gauss_spline)

    # Compare results
    np.testing.assert_allclose(cpu_gauss_spline, gpu_gauss_spline, 1e-3)    

    # Run multiple passes to get average
    for _ in range(loops):
        with prof.time_range("cupy_gauss_spline_loop", 2):
            gpu_gauss_spline = gauss_spline(d_x, d_n)