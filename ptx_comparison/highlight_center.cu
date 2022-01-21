#include <stdint.h>

struct pixel {
  uint16_t b;
  uint16_t g;
  uint16_t r;
  uint16_t a;
};

constexpr uint32_t image_width = 311;
constexpr uint32_t image_height = 324;

constexpr uint32_t center_x = image_width / 2;
constexpr uint32_t center_y = image_height / 2;
constexpr uint32_t radius = std::max(center_x, center_y);


extern "C"
__device__ void highlight_center(pixel& output, pixel pixel, uint32_t x, uint32_t y)
{
  uint32_t x_dist = std::abs((int)(x - center_x));
  uint32_t y_dist = std::abs((int)(y - center_y));
  double distance_from_center = std::sqrt((x_dist * x_dist) + (y_dist * y_dist));

  // Weight pixels according to distance from centre
  double w = 1 - (distance_from_center / radius);

  // Apply weight to each RGB component
  output.r = pixel.r * w;
  output.g = pixel.g * w;
  output.b = pixel.b * w;
  output.a = pixel.a;
}
