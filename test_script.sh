#!/bin/bash
declare LOOPS=$1
declare SCIPY=1
declare CUPY_RAW=1
declare CUPY_ELEMENT=2

# Test Scipy
for (( n=1; n<=${SCIPY}; n++ ))
do
	echo -e "**************************************************"
	echo -e "Test serial_python_v${n}.py ${LOOPS}"
	echo -e "**************************************************"
	nsys profile --sample=none --trace=cuda,nvtx --stats=true python3 serial_python/serial_python_v${n}.py ${LOOPS}
	echo -e
done

# Test CuPy Raw Kernel
for (( n=1; n<=${CUPY_RAW}; n++ ))
do
	echo -e "**************************************************"
	echo -e "Test raw_kernel_cupy_v${n}.py ${LOOPS}"
	echo -e "**************************************************"
	nsys profile --sample=none --trace=cuda,nvtx --stats=true python3 raw_kernel_cupy/raw_kernel_cupy_v${n}.py ${LOOPS}
	echo -e
done

# Test CuPy Elmentwise Kernels
for (( n=1; n<=${CUPY_ELEMENT}; n++ ))
do
	echo -e "**************************************************"
	echo -e "Test elementwise_kernel_cupy_v${n}.py ${LOOPS}"
	echo -e "**************************************************"
	nsys profile --sample=none --trace=cuda,nvtx --stats=true python3 elementwise_kernel_cupy/elementwise_kernel_cupy_v${n}.py ${LOOPS}
	echo -e
done

