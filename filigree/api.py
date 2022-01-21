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

# High-level Python API for the Filigree library

from PIL import Image as PILImage

from filigree import _lib, numba_extension


class Image:
    """
    Represents an on-device image. Construct an image from a file on-disk with:

        Image(filename)
    """
    def __init__(self, filename):
        self._lib_image = _lib.Image(filename)

    def apply_weighted_greyscale(self):
        """Convert the image to greyscale using Filigree's built-in kernel"""
        self._lib_image.apply_weighted_greyscale()

    def binarize(self):
        """Convert the image to black and white using Filigree's built-in kernel
        """
        self._lib_image.binarize()

    def to_pillow(self):
        """Copy the image back from the device and create a Pillow image from
        it."""
        data = self._lib_image.get_data()
        width = self._lib_image.get_width()
        height = self._lib_image.get_height()
        size = (width, height)
        pilimage = PILImage.frombytes("RGBA", size, data, "raw", "BGRA;16L")

        # Retaining the alpha channel seems to always make images completely
        # transparent (possibly due to something about PIL or images that I
        # don't understand, so the alpha channel is removed before returning
        # the image.
        return pilimage.convert("RGB")

    def apply_pixel_udf(self, udf):
        """Apply the given UDF to the image. `udf` should be a Python function
        that accepts a `pixel` and returns a tuple of red, green, blue, and
        alpha values. For example, a no-op UDF would look like:

            def noop(pixel):
                return pixel.r, pixel.g, pixel.b, pixel.a

        The intensity of each channel is an integer between 0 and 65535 (when
        the ImageMagick Quantum depth is 16, which it is generally expected to
        be).
        """
        numba_extension.apply_pixel_udf(self, udf)

    def apply_located_pixel_udf(self, udf):
        """Apply the given UDF to the image. `udf` should be a Python function
        that accepts a `pixel`, and the x and y coordinates of the pixel, and
        returns a tuple of red, green, blue, and alpha values. For example, a
        no-op located UDF would look like:

            def noop_located(pixel, x, y):
                return pixel.r, pixel.g, pixel.b, pixel.a

        The x and y coordinates are passed into the UDF so that it can
        spatially parameterize the transformation of the pixel - for example,
        to increase / decrease intensity depending on the location of the
        pixel.

        The intensity of each channel is an integer between 0 and 65535 (when
        the ImageMagick Quantum depth is 16, which it is generally expected to
        be).
        """

        numba_extension.apply_located_pixel_udf(self, udf)

    @property
    def pixel_ptr(self):
        return self._lib_image.get_pixel_ptr()

    @property
    def width(self):
        return self._lib_image.get_width()

    @property
    def height(self):
        return self._lib_image.get_height()
