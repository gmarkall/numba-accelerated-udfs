from filigree import Image
from numba.types import float32


image = Image('/home/gmarkall/tmp/timestamps.png')


def redize_udf(pixel):
    weighted = (
        float32(0.299) * pixel.r +
        float32(0.587) * pixel.g +
        float32(0.114) * pixel.b
    )
    return weighted, weighted, weighted, 1


image.apply_pixel_udf(redize_udf)
