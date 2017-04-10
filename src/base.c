/* basis program for image convolution, using naive scheme to compute for use as a benchmark to compare against */

/* standard headers */
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

/* external library headers */
#include <FreeImage.h>
#define CL_USE_DEPRECATED_OPENCL_1_2_APIS
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

  /* platforms and devices */
  cl_platform_id *platform_ids;
  cl_uint num_platforms;
  cl_device_id device_id;
  cl_uint num_devices;

  /* contexts and contexts specific variables */
  cl_context context;
  cl_command_queue commands;
  cl_program program;
  cl_kernel kernel;

  cl_int err;

  /* read source */
  char *kernel_source = NULL;
  read_cl_source("base.cl",&kernel_source);

  /* get all of the platforms */
  clGetPlatformIDs(0,NULL,&num_platforms);
  platform_ids = malloc(sizeof(cl_platform_id) * num_platforms);
  err = clGetPlatformIDs(num_platforms,platform_ids,&num_platforms);
  if(err){
    print_error("clGetPlatformIDs()",err);
    exit(EXIT_FAILURE);
  }

  /* get a device on the platforms which corresponds to the device type specified */
  for(unsigned int i = 0; i < num_platforms; ++i){
    clGetDeviceIDs(platform_ids[i],device_type,0,NULL,&num_devices);
    if(num_devices > 0){
      err = clGetDeviceIDs(platform_ids[i],device_type,1,&device_id,&num_devices);
      if(err != CL_SUCCESS || err != CL_DEVICE_NOT_FOUND){
        print_error("clGetDeviceIDs()",err);
        exit(EXIT_FAILURE);
      }
      /* no need for any other platform ids since we are only using 1 platform
       * so we reallocate
       */
      cl_platform_id temp = platform_ids[i];
      platform_ids = realloc(platform_ids,sizeof(cl_platform_id));
      *platform_ids = temp;
      break;
    }
  }

  /* create context */
  context = clCreateContext(platform_ids,1,&device_id,NULL,NULL,&err);
  if(err){
    print_error("clCreateContext()",err);
    exit(EXIT_FAILURE);
  }



  /* get the image */
  char * const image_path = argv[1];
  struct gimc_image image;

  /* load grayscale of image */
  gimc_image_load(&image,image_path);

  /* save output */
  FreeImage_Save(FIF_JPEG,image.bitmap,"gray.jpg",JPEG_DEFAULT);

  free(platform_ids);
  gimc_image_unload(&image);
  return 0;
}
