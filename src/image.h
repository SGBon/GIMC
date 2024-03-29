#ifndef GIMC_IMAGE_H
#define GIMC_IMAGE_H

#include <stdint.h>
#include <FreeImage.h>

struct gimc_image{
  FIBITMAP *bitmap;
  uint8_t *bits;
  size_t width;
  size_t height;
};

/* load an image file into a gimc_image struct */
extern void gimc_image_load(struct gimc_image *image,const char * filename);

/* free resources used by image */
extern void gimc_image_unload(struct gimc_image *image);

#endif
