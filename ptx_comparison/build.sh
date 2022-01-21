#!/bin/bash

NVCCFLAGS="--expt-relaxed-constexpr -gencode arch=compute_75,code=sm_75"

nvcc -ptx ${NVCCFLAGS} ptx_comparison.cu -o cpp.ptx
# RDC needed to prevent the device function being optimized away
nvcc -rdc true -ptx ${NVCCFLAGS} highlight_center.cu -o highlight_cpp.ptx
python ptx_comparison.py > python.ptx
python highlight_center.py > highlight_python.ptx
