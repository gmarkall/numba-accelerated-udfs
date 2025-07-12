#!/bin/bash

set -euo pipefail

export PKG_CONFIG_PATH=${PKG_CONFIG_PATH:-}:$CONDA_PREFIX/lib/pkgconfig

build_im()
{
  echo "Building ImageMagick..."
  mkdir im_build
  cd im_build
  ../vendor/ImageMagick6/configure --prefix=$CONDA_PREFIX
  make -j12
  make install
  cd ..
}

build_pillow()
{
  echo "Building Pillow..."
  cd vendor/Pillow
  python setup.py develop
  cd ../..
}

build_filigree_cpp()
{
  echo "Building Filigree C++ components..."
  mkdir build
  cd build
  cmake .. -G Ninja -DCMAKE_INSTALL_PREFIX=$CONDA_PREFIX
  ninja
  ninja install
  cd ..
}

build_filigree_python()
{
  echo "Building Filigree Python components..."
  python setup.py develop
}

build_im
build_pillow
build_filigree_cpp
build_filigree_python
