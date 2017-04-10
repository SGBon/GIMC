#include "image.h"

void gimc_image_load(struct gimc_image *image,const char * filename){
  FIBITMAP *bitmap;

  /* load image and convert to greyscale */
  const FREE_IMAGE_FORMAT fif = FreeImage_GetFIFFromFilename(filename);
  bitmap = FreeImage_Load(fif, filename,0);
  image->bitmap = FreeImage_ConvertToGreyscale(bitmap);
  FreeImage_Unload(bitmap);

  image->width = FreeImage_GetWidth(image->bitmap);
  image->height = FreeImage_GetHeight(image->bitmap);
  image->bits = FreeImage_GetBits(image->bitmap);
}

void gimc_image_unload(struct gimc_image *image){
  FreeImage_Unload(image->bitmap);
}
