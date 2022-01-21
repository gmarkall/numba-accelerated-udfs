from numba import cuda, uint32
from filigree.numba_extension import pixel_packet
import math

image_width = 311
image_height = 324

center_x = image_width // 2
center_y = image_height // 2
radius = max(center_x, center_y)


def highlight_center(pixel, x, y):
    distance_from_center = math.sqrt(
        abs(x - center_x)**2 + abs(y - center_y)**2)

    # Weight pixels according to distance from centre
    w = 1 - (distance_from_center / radius)

    # Apply weight to each RGB component
    return pixel.r * w, pixel.g * w, pixel.b * w, pixel.a


args = (pixel_packet, uint32, uint32)
ptx, resty = cuda.compile_ptx(highlight_center, args, cc=(7, 5), device=True)


# Shorten ptx name
fname = [line for line in ptx.split('\n')
         if 'globl' in line][0].split()[-1].strip()

print(ptx.replace(fname, 'highlight_center_py'))
