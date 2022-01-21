#include <stdint.h>


struct pixel {
  uint16_t b;
  uint16_t g;
  uint16_t r;
  uint16_t a;
};

__device__ void red_filter(pixel& output, pixel input)
{
  output.r = input.r;
  output.g = 0;
  output.b = 0;
  output.a = input.a;
}

extern "C"
__global__ void my_transform(pixel* image, size_t width, size_t height)
{
  uint32_t x = blockIdx.x * blockDim.x + threadIdx.x;
  uint32_t y = blockIdx.y * blockDim.y + threadIdx.y;

  if ((x > width) || (y > height))
    return;

  size_t idx = (y * width) + x;

  pixel input = image[idx];
  pixel output;
  red_filter(output, input);
  image[idx] = output; 
}
