from numba import cuda, void, int16, uint32
from filigree.numba_extension import filigree_image, filigree_image_handler


@cuda.jit
def red_filter(pixel):
    # Strip out green and blue components
    return pixel.r, int16(0), int16(0), pixel.a


@cuda.jit(void(filigree_image), extensions=[filigree_image_handler])
def my_transform(image):
    x, y = cuda.grid(2)
    x = uint32(x)
    y = uint32(y)

    if x >= image.width or y >= image.height:
        return

    pixel = image[x, y]
    r, g, b, a = red_filter(pixel.values)
    pixel.r = r
    pixel.g = g
    pixel.b = b
    pixel.a = a


ptx = next(iter(my_transform.inspect_asm().values()))

# Shorten ptx name
fname = [line for line in ptx.split('\n')
         if 'globl' in line][0].split()[-1].strip()

print(ptx.replace(fname, 'my_transform'))
