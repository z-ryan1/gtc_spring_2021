import cupy as cp
import numpy as np
import sys

from cupy import prof
from scipy import signal
from string import Template

# Raw kernel implementation of CuPy

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

        const double PI = 3.14159265358979;
        for ( int tid = tx; tid < x_shape; tid += stride) {
            double x_val { x[tid] };
            double signsq {};
            double r_signsq {};
            double res1 {};
            
            signsq = ( 1.0 + n ) / 12.0;
            r_signsq = 0.5 / signsq;
            res1 = ( 1.0 / sqrt( 2.0 * PI * signsq )) * exp( -( x_val * x_val ) * r_signsq);
            res[tid] = (
                res1
            );
        }

    }
}
"""
)


def _gauss_spline(x, n, res):

    device_id = cp.cuda.Device()
    numSM = device_id.attributes["MultiProcessorCount"]
    threadsperblock = (128,)
    blockspergrid = (numSM * 20,)

    src = _cupy_gauss_spline_src.substitute(datatype="double")
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
    res = cp.empty(x.shape[0], dtype="double")
    _gauss_spline(x, n, res)

    return res


def rand_data_gen_gpu(num_samps, dim=1, dtype=np.float64):
    inp = tuple(np.ones(dim, dtype=int) * num_samps)
    cpu_sig = np.random.random(inp)
    cpu_sig = cpu_sig.astype(dtype)
    gpu_sig = cp.asarray(cpu_sig)

    return cpu_sig, gpu_sig


def main():
    loops = int(sys.argv[1])

    n = np.random.randint(0, 1234)

    num_samps = 2 ** 16
    x, y = rand_data_gen_gpu(num_samps)

    # Run baseline with scipy.signal.gauss_spline
    with prof.time_range("scipy_gauss_spline", 0):
        cpu_gauss_spline = signal.gauss_spline(x, n)

    # Run CuPy version
    with prof.time_range("cupy_gauss_spline", 1):
        gpu_gauss_spline = gauss_spline(y, n)

    # Compare results
    np.testing.assert_allclose(
        cpu_gauss_spline, cp.asnumpy(gpu_gauss_spline), 1e-3
    )

    # Run multiple passes to get average
    for _ in range(loops):
        with prof.time_range("cupy_gauss_spline_loop", 2):
            gpu_gauss_spline = gauss_spline(y, n)


if __name__ == "__main__":
    sys.exit(main())
