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

# Numba extension to support UDFs in Filigree

from numba import cuda, types
from numba.core import cgutils
from numba.core.extending import (make_attribute_wrapper, models,
                                  register_model, typeof_impl)
from numba.core.typing import signature
from numba.core.typing.templates import AttributeTemplate, CallableTemplate
from numba.cuda.cudadecl import registry as cuda_registry
from numba.cuda.cudaimpl import (lower as cuda_lower, lower_attr as
                                 cuda_lower_attr, registry as
                                 cuda_lower_registry)

from filigree import _lib, api

import operator


# A utility function to get the ImageMagick Quantum depth. This is used to
# ensure that the data structures used in the Numba extension match those used
# in the C++ Filigree library.
def get_quantum():
    bitwidth = _lib.get_quantum_depth()
    return {8: types.uint8, 16: types.uint16}[bitwidth]


QUANTUM = get_quantum()


# ------------------------------------------------------------------------------
# Typing
# ------------------------------------------------------------------------------

# The Numba type for a PixelPacket. This mirrors the Magick::PixelPacket type
# in C++.
class PixelPacket(types.Type):
    def __init__(self):
        super().__init__(name="PixelPacket")


# Often typing in Numba uses an instance of a type rather than the type class,
# so we instantiate one for general use here.
pixel_packet = PixelPacket()


# Implement typing for each of the attributes of a PixelPacket. This informs
# Numba that e.g. accessing `pp.r` for a PixelPacket `pp` will provide a value
# of type `QUANTUM` (and similar for the other channels).
class PixelPacketAttrsTemplate(AttributeTemplate):

    def resolve_r(self, pp):
        return QUANTUM

    def resolve_g(self, pp):
        return QUANTUM

    def resolve_b(self, pp):
        return QUANTUM

    def resolve_a(self, pp):
        return QUANTUM


# Register the typing for PixelPacket attributes with Numba.
@cuda_registry.register_attr
class PixelPacketAttrs(PixelPacketAttrsTemplate):
    key = pixel_packet


# Represents a pointer to a PixelPacket. We implement this type to allow us to
# pass around pointers to image data and modify it in-place through the
# pointer.
class PixelPacketPointer(types.RawPointer):
    def __init__(self):
        super().__init__(name="PixelPacket*")


# Inastance of the PixelPacket pointer for general use.
pixel_packet_pointer = PixelPacketPointer()


# Implement typing for the attribute `ppp.values` of a PixelPacketPointer
# `ppp` - it returns the values, a PixelPacket. This is in addition to the
# channels `r`, `g`, etc., which we inheried from the PixelPacketAttrsTemplate.
@cuda_registry.register_attr
class PixelPacketPointerAttrs(PixelPacketAttrsTemplate):
    key = pixel_packet_pointer

    def resolve_values(self, pp):
        return pixel_packet


# A type for the Filigree Image class itself.
class FiligreeImage(types.Type):
    def __init__(self):
        super().__init__(name="FiligreeImage")


# Filigree image instance for general use.
filigree_image = FiligreeImage()


# This enables Numba to recognise instances of the Python api.Image class as
# being of the FiligreeImage type when they are passed into @cuda.jit kernels.
# This information is needed so that Numba knows how to prepare the argument,
# and to seed the type inference process.
@typeof_impl.register(api.Image)
def typeof_filigree_image(val, c):
    return filigree_image


# Typing for getitem on an Image: `img[x, y]` returns a PixelPacketPointer.
@cuda_registry.register_global(operator.getitem)
class Image_getitem(CallableTemplate):
    def generic(self):
        def typer(image, indices):
            if not isinstance(image, FiligreeImage):
                return None
            if not isinstance(indices, types.BaseTuple) or len(indices) != 2:
                return None
            return signature(pixel_packet_pointer, image, indices)

        return typer


# ------------------------------------------------------------------------------
# Data models
# ------------------------------------------------------------------------------

# Data model for the PixelPacketPointer. Since we mainly treat this as an
# opaque pointer when passing it around, we use the OpaqueModel from Numba.
@register_model(PixelPacketPointer)
class PixelPacketPointerModel(models.OpaqueModel):
    pass


# Data model for PixelPackets. Since we work with the data for a PixelPacket,
# we use a StructModel that enables Numba to interpret the data underlying a
# PixelPacket.
@register_model(PixelPacket)
class PixelPacketModel(models.StructModel):
    def __init__(self, dmm, fe_type):
        # The order and type of these must match the ImageMagick::PixelPacket
        # struct, which uses the BGRA format.
        members = [
            ("b", QUANTUM),
            ("g", QUANTUM),
            ("r", QUANTUM),
            ("a", QUANTUM)
        ]
        super().__init__(dmm, fe_type, members)


# Data model for an Image. As discussed in the corresponding presentation, this
# doesn't match the structure of the Image class in C++, but instead Numba
# prepares this structure using an argument handling extension when an Image is
# passed to a kernel.
@register_model(FiligreeImage)
class FiligreeImageModel(models.StructModel):
    def __init__(self, dmm, fe_type):
        members = [
            ("pixel_ptr", pixel_packet_pointer),
            ("width", types.uint64),
            ("height", types.uint64),
        ]
        super().__init__(dmm, fe_type, members)


# ------------------------------------------------------------------------------
# Lowering
# ------------------------------------------------------------------------------

# A lowering "shortcut" that allows each property of the PixelPacket in Python
# code to be turned into an access to the struct member in the LLVM IR.
make_attribute_wrapper(PixelPacket, 'r', 'r')
make_attribute_wrapper(PixelPacket, 'g', 'g')
make_attribute_wrapper(PixelPacket, 'b', 'b')
make_attribute_wrapper(PixelPacket, 'a', 'a')


# Lowering shortcuts for Image width and height.
make_attribute_wrapper(FiligreeImage, 'width', 'width')
make_attribute_wrapper(FiligreeImage, 'height', 'height')


# Lowering for `img[x, y]`.
@cuda_lower(operator.getitem, filigree_image, types.UniTuple)
def filigree_image_getitem(context, builder, sig, args):
    # Unpack the arguments and argument types
    image_arg, index_arg = args
    image_ty, index_ty = sig.args
    # Create a struct proxy to conveniently generate code to access members of
    # the Image struct's data.
    image = cgutils.create_struct_proxy(image_ty)(context, builder,
                                                  value=image_arg)

    # Unpack the indices
    x, y = cgutils.unpack_tuple(builder, index_arg, count=2)

    # Ensure indices are of the same type by casting them up to 64 bits
    x = context.cast(builder, x, index_ty[0], types.uint64)
    y = context.cast(builder, y, index_ty[1], types.uint64)
    offset = builder.add(builder.mul(y, image.width), x)

    # Cast the image's pixel packet pointer so that pointer arithmetic
    # functions as expected
    ptr = builder.bitcast(image.pixel_ptr,
                          context.get_value_type(pixel_packet).as_pointer())

    # Determine the address of the pixel at the point x, y and return it
    current_pixel = cgutils.gep_inbounds(builder, ptr, offset)
    return current_pixel


# Lowering for `ppp.values` - returns the struct of r, g, b, a, by loading from
# the pointer.
@cuda_lower_attr(pixel_packet_pointer, 'values')
def pixel_packet_pointer_values(context, builder, sig, arg):
    return builder.load(arg)


# Lowering for `ppp.<attr> = val`. We use lower_setattr_generic so that we only
# need write one lowering function for each of r, g, b, and a. The attribute
# name is passed into the lowering function in the `attr` parameter.
#
# Although the generated IR loads all channels, updates a single channel, and
# stores all channels, it is expected that LLVM will optimize away the loads /
# stores of the individual unmodified channels.
@cuda_lower_registry.lower_setattr_generic(pixel_packet_pointer)
def pixel_packet_pointer_set_attr(context, builder, sig, args, attr):
    # Get a pointer to the values to be updated
    base_idx = context.get_constant(types.intp, 0)
    value_ptr = cgutils.gep_inbounds(builder, args[0], base_idx)

    # Load the existing struct values into a struct proxy
    data = builder.load(value_ptr)
    values = cgutils.create_struct_proxy(pixel_packet)(context, builder,
                                                       value=data)

    # Update the struct proxy for the given channel
    setattr(values, attr, args[1])

    # Write the modified struct back to memory
    builder.store(values._getvalue(), value_ptr)


# ------------------------------------------------------------------------------
# Image argument extension handling
# ------------------------------------------------------------------------------

# The image argument is passed as a tuple of three uint64s, which matches the
# layout of the data model for an Image.
image_args_type = types.UniTuple(types.uint64, 3)


# This argument handler converts Image instances into tuples of their pointer,
# width, and height (the `image_args_type` above).
class FiligreeImageHandler:
    def prepare_args(self, ty, val, **kwargs):
        if isinstance(val, api.Image):
            args = (val.pixel_ptr, val.width, val.height)
            return image_args_type, args
        else:
            return ty, val


# An instance of the argument handler is needed by the extensions kwarg of
# @cuda.jit decorators, so we instantiate one here.
filigree_image_handler = FiligreeImageHandler()


# ------------------------------------------------------------------------------
# Launch API
# ------------------------------------------------------------------------------

# Both the launch functions here work similarly. They create a new @cuda.jit
# kernel that closes over the UDF, and launch it. The kernel handles mapping
# the thread ID and loading / storing of pixel data for UDFs, so that they can
# be written independently of any CUDA abstractions.


# Applies a UDF that transforms pixels independent of their location.
# Passed-in UDFs should accept a PixelPacket and return a tuple of r, g, b, and
# a values.
def apply_pixel_udf(image, udf):
    device_function = cuda.jit(device=True)(udf)

    @cuda.jit(extensions=[filigree_image_handler])
    def apply_udf(image):
        x, y = cuda.grid(2)

        if x < image.width and y < image.height:
            pixel_ptr = image[x, y]
            r, g, b, a = device_function(pixel_ptr.values)
            pixel_ptr.r = r
            pixel_ptr.g = g
            pixel_ptr.b = b
            pixel_ptr.a = a

    nthreads_x = 16
    nthreads_y = 8
    nblocks_x = (image.width // nthreads_x) + 1
    nblocks_y = (image.height // nthreads_y) + 1
    grid_dim = (nblocks_x, nblocks_y)
    block_dim = (nthreads_x, nthreads_y)

    apply_udf[grid_dim, block_dim](image)


# Applies a UDF that transforms pixels dependent on their location.
# Passed-in UDFs should accept a PixelPacket, and the x and y coordinates of
# the image,  and return a tuple of r, g, b, and a values.
def apply_located_pixel_udf(image, udf):
    device_function = cuda.jit(device=True)(udf)

    @cuda.jit(extensions=[filigree_image_handler])
    def apply_udf(image):
        x, y = cuda.grid(2)

        if x < image.width and y < image.height:
            pixel_ptr = image[x, y]
            r, g, b, a = device_function(pixel_ptr.values, x, y)
            pixel_ptr.r = r
            pixel_ptr.g = g
            pixel_ptr.b = b
            pixel_ptr.a = a

    nthreads_x = 16
    nthreads_y = 8
    nblocks_x = (image.width // nthreads_x) + 1
    nblocks_y = (image.height // nthreads_y) + 1
    grid_dim = (nblocks_x, nblocks_y)
    block_dim = (nthreads_x, nthreads_y)

    apply_udf[grid_dim, block_dim](image)
