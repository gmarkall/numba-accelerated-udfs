import os
import pathlib

CONDA_PREFIX = os.environ['CONDA_PREFIX']
REPO = pathlib.Path(__file__).parent.resolve()

flags = [
    f'-I{CONDA_PREFIX}/include/python3.10',
    f'-I{CONDA_PREFIX}/include',
    f'-I{CONDA_PREFIX}/lib/python3.10/site-packages/numpy/core/include',
    f'-I{CONDA_PREFIX}/include/ImageMagick-6',
    f'-I{REPO}/build/_deps/argparse-src/include',
    f'-I{REPO}/build/_deps/rmm-src/include',
    f'-I{REPO}/build/_deps/spdlog-src/include',
    f'-I{REPO}/include',
    '-I/usr/local/cuda/include',
    '-DMAGICKCORE_HDRI_ENABLE=0',
    '-DMAGICKCORE_QUANTUM_DEPTH=16',
    '--cuda-gpu-arch=sm_50',
    '--std=c++17'
]


def Settings(**kwargs):
    return { 'flags': flags }
