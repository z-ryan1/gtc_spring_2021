#!/bin/bash
declare LOOPS=$1

if [ -z "$LOOPS" ];
then
	LOOPS=10
else
	LOOPS=${LOOPS}
fi

#Test Scipy

echo -e "**************************************************"
echo -e "Test serial_python_v1.py ${LOOPS}"
echo -e "**************************************************"
nsys profile --sample=none --trace=cuda,nvtx --stats=true python3 serial_python_v1.py ${LOOPS}
echo -e


# Test CuPy Raw Kernel

echo -e "**************************************************"
echo -e "Test raw_kernel_cupy_v1.py ${LOOPS}"
echo -e "**************************************************"
nsys profile --sample=none --trace=cuda,nvtx --stats=true python3 raw_kernel_cupy_v1.py ${LOOPS}
echo -e

# Test CuPy Elementwise Kernel - Gauss Spline

echo -e "**************************************************"
echo -e "Test elementwise_kernel_cupy_v1.py ${LOOPS}"
echo -e "**************************************************"
nsys profile --sample=none --trace=cuda,nvtx --stats=true python3 elementwise_kernel_cupy_v1.py ${LOOPS}
echo -e

# Test CuPy Elementwise Kernel - Signal components 

echo -e "**************************************************"
echo -e "Test elementwise_kernel_cupy_v2.py ${LOOPS}"
echo -e "**************************************************"
nsys profile --sample=none --trace=cuda,nvtx --stats=true python3 elementwise_kernel_cupy_v2.py ${LOOPS}
echo -e