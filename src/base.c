/* standard headers */
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

/* external library headers */
#include <FreeImage.h>
#include <CL/cl.h>

/* project headers */
#include "common.h"

int main(int argc, char **argv){
  FIBITMAP *bitmap;
  struct gimc_image image;
  char * const image_file = argv[1];

  /* load image and convert to greyscale */
  const FREE_IMAGE_FORMAT fif = FreeImage_GetFIFFromFilename(image_file);
  bitmap = FreeImage_Load(fif, image_file,0);
  image.bitmap = FreeImage_ConvertToGreyscale(bitmap);
  FreeImage_Unload(bitmap);

  image.width = FreeImage_GetWidth(image.bitmap);
  image.height = FreeImage_GetHeight(image.bitmap);
  const unsigned int depth = FreeImage_GetBPP(image.bitmap)/8;
  const size_t image_size = image.width * image.height * depth;

  uint8_t * const image_data = FreeImage_GetBits(image.bitmap);

  FreeImage_Save(FIF_JPEG,image.bitmap,"gray.jpg",JPEG_DEFAULT);

  FreeImage_Unload(image.bitmap);
  return 0;
}
