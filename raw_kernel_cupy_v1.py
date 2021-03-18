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
            const double * __restrict__ x,
            const int n,
            double * __restrict__ res
            ) {
            
        const int tx { static_cast<int>( blockIdx.x * blockDim.x + threadIdx.x ) };
        const int stride { static_cast<int>( blockDim.x * gridDim.x ) };

        const double PI = 3.14159265358979f;
        for ( int tid = tx; tid < x_shape; tid += stride) {
            double x_val { x[tid] };
            double signsq {};
            double val {};
            double val2 {};
            double val3 {};
            double val4 {};
            double res1 {};
            
            signsq = (1.0 + n) / 12.0;
            val = 2 * signsq * PI;
            val2 = 2.0 / 2.0 / signsq;
            val3 = pow(-x_val, val2 );
            val4 = exp( val3 );
            res1 = 1.0 / sqrt(val) * val4;
            res[tid] = (
                1
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

    #x = [ 2 ** 16 ]
    x = [ 0, 1, 1]
    n = 1
    in_samps = 2 ** 10

    #n = np.random.randint(0, 1234)
    #x = np.linspace(0.01, 10 * np.pi, in_samps)

    

    # Run baseline with scipy.signal.gauss_spline
    with prof.time_range("scipy_gauss_spline", 0):
        cpu_gauss_spline = signal.gauss_spline(x, n)

    d_x = cp.array(x)

    # Run CuPy version
    with prof.time_range("cupy_gauss_spline", 1):
        gpu_gauss_spline = gauss_spline(d_x, n)
        print(gpu_gauss_spline)

    # Compare results
    np.testing.assert_allclose(cpu_gauss_spline, cp.asnumpy(gpu_gauss_spline), 1e-3)    

    # Run multiple passes to get average
    for _ in range(loops):
        with prof.time_range("cupy_gauss_spline_loop", 2):
            gpu_gauss_spline = gauss_spline(d_x, n)