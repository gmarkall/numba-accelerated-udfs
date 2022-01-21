# =============================================================================
# Copyright (c) 2022, NVIDIA CORPORATION.
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not
# use this file except in compliance with the License. You may obtain a copy of
# the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
# WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
# License for the specific language governing permissions and limitations under
# the License.
# =============================================================================

cmake_minimum_required(VERSION 3.20.1 FATAL_ERROR)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

file(DOWNLOAD https://raw.githubusercontent.com/rapidsai/rapids-cmake/branch-22.02/RAPIDS.cmake
  ${CMAKE_BINARY_DIR}/RAPIDS.cmake)
include(${CMAKE_BINARY_DIR}/RAPIDS.cmake)
include(rapids-cmake)
include(rapids-cpm)
include(rapids-cuda)
include(rapids-export)
include(rapids-find)

rapids_cuda_init_architectures(FILIGREE)

project(
  FILIGREE
  VERSION 22.03
  LANGUAGES C CXX CUDA
)

# Set a default build type if none was specified
rapids_cmake_build_type(Debug)

# find packages we depend on
rapids_find_package(
  CUDAToolkit REQUIRED
)
rapids_cpm_init()

find_package(ImageMagick REQUIRED COMPONENTS Magick++)
include_directories(${ImageMagick_INCLUDE_DIRS})

# Add necessary defines for ImageMagick - see CMake Issue #22637:
# https://gitlab.kitware.com/cmake/cmake/-/issues/22637
pkg_check_modules(IM "ImageMagick")
add_compile_options(${IM_CFLAGS_OTHER})

include(cmake/thirdparty/get_rmm.cmake)

CPMAddPackage(
  NAME argparse
  GITHUB_REPOSITORY p-ranav/argparse
  GIT_TAG v2.2
  DOWNLOAD_ONLY
)

if(argparse_ADDED)
  include_directories("${argparse_SOURCE_DIR}/include")
endif()

include_directories("${CMAKE_SOURCE_DIR}/include")

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

add_subdirectory(src)
