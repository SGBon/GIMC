#ifndef GIMC_COMMON_H
#define GIMC_COMMON_H

#include <stdint.h>
#include <FreeImage.h>

struct gimc_image{
  FIBITMAP *bitmap;
  size_t width;
  size_t height;
};

#endif
