#!/bin/bash
declare LOOPS=$1

# Test Scipy

echo -e "**************************************************"
echo -e "Test serial_python_v${n}.py ${LOOPS}"
echo -e "**************************************************"
nsys profile --sample=none --trace=cuda,nvtx --stats=true python3 serial_python_v${n}.py ${LOOPS}
echo -e


# Test CuPy Raw Kernel

echo -e "**************************************************"
echo -e "Test raw_kernel_cupy_v${n}.py ${LOOPS}"
echo -e "**************************************************"
nsys profile --sample=none --trace=cuda,nvtx --stats=true python3 raw_kernel_cupy_v${n}.py ${LOOPS}
echo -e

# Test CuPy Elmentwise Kernels

echo -e "**************************************************"
echo -e "Test elementwise_kernel_cupy_v${n}.py ${LOOPS}"
echo -e "**************************************************"
nsys profile --sample=none --trace=cuda,nvtx --stats=true python3 elementwise_kernel_cupy_v${n}.py ${LOOPS}
echo -e

