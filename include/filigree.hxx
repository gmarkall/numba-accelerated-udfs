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

#include <string>
#include <Magick++.h>
#include <rmm/cuda_stream.hpp>
#include <rmm/mr/device/cuda_memory_resource.hpp>

namespace filigree {

/* The Filigree Image object.
 *
 * Images are represented by a buffer of ImageMagick PixelPackets held on the
 * GPU. Memory management is handled by the RAPIDS Memory Manager (RMM).
 */

class Image
{
  public:
    /* Consruct an image by loading data from a file. The memory resource and
     * stream are used to allocate space on the device. */
    Image(std::string filename,
          rmm::mr::device_memory_resource *mr,
          rmm::cuda_stream_view stream);

    /* Dtor deallocates the image buffer. */
    ~Image();

    /* Functions to launch kernels that transform the image. */
    void to_greyscale();
    void to_binary();

    /* Write the image buffer to a the given file. */
    void write(std::string filename);

    /* Copies the data to the host and returns a pointer to the data. */
    Magick::PixelPacket* get_data();

    /* Get the pointer to the data on the device. */
    Magick::PixelPacket* get_pixel_ptr();

    /* Accessors for image dimensions and allocation size. */
    size_t get_alloc_size();
    size_t get_width();
    size_t get_height();

  private:
    /* A utility function to deallocate the image buffer. */
    void cleanup();

    /* The RMM memory resource and the stream on which it operates. The image
     * buffer is allocated from the memory resource's pool. */
    rmm::mr::device_memory_resource *_mr;
    rmm::cuda_stream_view _stream;

    /* A pointer to the image buffer. */
    Magick::PixelPacket *_pixels;

    /* Dimensions of the image. */
    size_t _width;
    size_t _height;

    /* The size of the buffer in bytes. */
    size_t _alloc_size;
};

} // namespace filigree
