# Copyright (c) 2022, NVIDIA CORPORATION.
#
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

import os
import subprocess
from setuptools import setup, Extension

base_dir = os.path.dirname(os.path.abspath(__file__))
build_dir = os.path.join(base_dir, 'build')

filigree_include = os.path.join(base_dir, 'include')
cuda_include = os.path.join(os.sep, 'usr', 'local', 'cuda-11.5', 'include')
rmm_include = os.path.join(build_dir, '_deps', 'rmm-src', 'include')
spdlog_include = os.path.join(build_dir, '_deps', 'spdlog-src', 'include')

include_dirs = [filigree_include, cuda_include, rmm_include, spdlog_include]


def get_im_cflags():
    im_pkg_config_cmd = "pkg-config --cflags Magick++"
    cp = subprocess.run(im_pkg_config_cmd, shell=True, check=True,
                        capture_output=True)
    return cp.stdout.decode().strip().split()


extra_compile_args = ['-Wall', '-Werror'] + get_im_cflags()

module = Extension(
    'filigree._lib',
    sources=['filigree/_lib.cpp'],
    include_dirs=include_dirs,
    libraries=['filigree'],
    extra_compile_args=extra_compile_args,
)

setup(
    name='filigree',
    version='0.1',
    description='A simple customisable image processing library for CUDA',
    ext_modules=[module],
    packages=['filigree'],
)
