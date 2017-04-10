/* standard headers */
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

/* external library headers */
#include <FreeImage.h>
#include <CL/cl.h>

/* project headers */
#include "image.h"
#include "clutil.h"

int main(int argc, char **argv){
  if(argc < 3){
    printf("Usage: %s [Image File] [Device Option]\n",argv[0]);
    return -1;
  }

  cl_device_type device_type;
  switch(atoi(argv[2])){
  case 0:
    device_type = CL_DEVICE_TYPE_CPU;
    break;
  case 1:
  default:
    device_type = CL_DEVICE_TYPE_GPU;
    break;
  }

  char * const image_path = argv[1];
  struct gimc_image image;

  /* load grayscale of image */
  gimc_image_load(&image,image_path);

  FreeImage_Save(FIF_JPEG,image.bitmap,"gray.jpg",JPEG_DEFAULT);

  gimc_image_unload(&image);
  return 0;
}
