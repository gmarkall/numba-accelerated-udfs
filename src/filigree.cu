/* Copyright (c) 2022, NVIDIA CORPORATION.

 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

/* The Filigree C++ library. */


#include <iostream>

#include "filigree.hxx"
#include "rmm/mr/device/per_device_resource.hpp"

namespace filigree {

/* Convert RGB values to an intensity value using the weighting:
 *
 *     0.299R + 0.587G + 0.114B
 */
__device__ Magick::Quantum weighted(Magick::PixelPacket *pixel)
{
  float weighted = 0.299f * pixel->red + 0.587f * pixel->green + 0.114 * pixel->blue;
  return static_cast<Magick::Quantum>(weighted);
}

/* Kernel that converts a color image to greyscale using the weighted method. */
__global__ void to_greyscale_kernel(Magick::PixelPacket *pixels, size_t width,
                                    size_t height) {
  size_t x = blockIdx.x * blockDim.x + threadIdx.x;
  size_t y = blockIdx.y * blockDim.y + threadIdx.y;

  if ((x < width) && (y < height)) {
    Magick::PixelPacket *pixel = &pixels[y * width + x];
    Magick::Quantum intensity = weighted(pixel);
    pixel->red = intensity;
    pixel->green = intensity;
    pixel->blue = intensity;
  }
}

/* Kernel that converts a color image to black and white - a pixel is made white
 * if its intensity is greater than 0.5. */
__global__ void binarize_kernel(Magick::PixelPacket *pixels, size_t width,
                                size_t height) {
  size_t x = blockIdx.x * blockDim.x + threadIdx.x;
  size_t y = blockIdx.y * blockDim.y + threadIdx.y;

  if ((x < width) && (y < height)) {
    Magick::PixelPacket *pixel = &pixels[y * width + x];
    float intensity = weighted(pixel);
    Magick::Quantum binary = (intensity > 32767) ? 65535 : 0;
    pixel->red = binary;
    pixel->green = binary;
    pixel->blue = binary;
  }
}

/* Construct an image by loading data from a file. ImageMagick is used to read
 * the file, which is then transferred to the device. */
Image::Image(std::string filename, rmm::mr::device_memory_resource *mr,
             rmm::cuda_stream_view stream)
    : _mr(mr), _stream(stream) {
  Magick::Image image;
  image.read(filename);

  _width = image.columns();
  _height = image.rows();

  Magick::Pixels cache(image);
  Magick::PixelPacket *pixels = cache.get(0, 0, _width, _height);

  _alloc_size = _width * _height * sizeof(Magick::PixelPacket);
  _pixels = (Magick::PixelPacket *)_mr->allocate(_alloc_size, _stream);

  auto err = cudaMemcpy(_pixels, pixels, _alloc_size, cudaMemcpyHostToDevice);
  if (err != cudaSuccess) {
    cleanup();
    throw std::runtime_error(
        "Copying image to device failed: cudaMemcpy failed");
  }

  /* Ensure that the DMA from staging memory to the final destination has
   * completed before returning (see "Synchronous: 2." in
   * https://docs.nvidia.com/cuda/cuda-runtime-api/api-sync-behavior.html#api-sync-behavior__memcpy-sync).
   */
  cudaDeviceSynchronize();
}

/* Launch the greyscale conversion kernel to transform the image to greyscale
 * in-place. */
void Image::to_greyscale() {
  unsigned int nthreads_x = 16;
  unsigned int nthreads_y = 8;
  unsigned int nblocks_x = (_width / nthreads_x) + 1;
  unsigned int nblocks_y = (_height / nthreads_y) + 1;

  dim3 nthreads{nthreads_x, nthreads_y};
  dim3 nblocks{nblocks_x, nblocks_y};

  to_greyscale_kernel<<<nblocks, nthreads>>>(_pixels, _width, _height);
}

/* Launch the binary conversion kernel to transform the image to greyscale
 * in-place. */
void Image::to_binary() {
  unsigned int nthreads_x = 16;
  unsigned int nthreads_y = 8;
  unsigned int nblocks_x = (_width / nthreads_x) + 1;
  unsigned int nblocks_y = (_width / nthreads_y) + 1;

  dim3 nthreads{nthreads_x, nthreads_y};
  dim3 nblocks{nblocks_x, nblocks_y};

  binarize_kernel<<<nblocks, nthreads>>>(_pixels, _width, _height);
}

/* Copies the data to the host and returns a pointer to the data. */
Magick::PixelPacket *Image::get_data() {
  Magick::PixelPacket *pixels = new Magick::PixelPacket[_alloc_size];
  auto err = cudaMemcpy(pixels, _pixels, _alloc_size, cudaMemcpyDeviceToHost);
  if (err != cudaSuccess) {
    throw std::runtime_error("Copying image to host failed: cudaMemcpy failed");
  }

  return pixels;
}

/* Get the pointer to the data on the device. */
Magick::PixelPacket *Image::get_pixel_ptr() { return _pixels; }

size_t Image::get_alloc_size() { return _alloc_size; }

size_t Image::get_width() { return _width; }

size_t Image::get_height() { return _height; }

/* Write the image buffer to a the given file. ImageMagick is used to write the
 * file. */
void Image::write(std::string filename) {
  Magick::Image image(Magick::Geometry(_width, _height),
                      Magick::Color("white"));
  image.modifyImage();
  Magick::Pixels cache(image);
  Magick::PixelPacket *cached_pixels = cache.get(0, 0, _width, _height);

  auto err =
      cudaMemcpy(cached_pixels, _pixels, _alloc_size, cudaMemcpyDeviceToHost);
  if (err != cudaSuccess) {
    throw std::runtime_error("Copying image to host failed: cudaMemcpy failed");
  }
  cache.sync();
  image.write(filename);
}

/* A utility function to deallocate the image buffer. */
void Image::cleanup() { _mr->deallocate(_pixels, _alloc_size, _stream); }

Image::~Image() { cleanup(); }

} // namespace filigree
