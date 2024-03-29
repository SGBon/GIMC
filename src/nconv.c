/* augmented basis program for convolution
 * performs n convolutions based on command line arguments
 */


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
#include "filter.h"

int main(int argc, char **argv){
  if(argc < 5){
    printf("Usage: %s [Image File] [Device Option] [Number of Filters] [Size of Filters]\n",argv[0]);
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

  /* get the image */
  char * const image_path = argv[1];
  struct gimc_image image;

  /* load grayscale of image */
  gimc_image_load(&image,image_path);

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

  /* variable for cl errors */
  cl_int err;

  cl_mem d_image;
  cl_mem d_filter;
  cl_mem d_result;

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
      if(err != CL_SUCCESS && err != CL_DEVICE_NOT_FOUND){
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
  context = clCreateContext(NULL,1,&device_id,NULL,NULL,&err);
  if(err){
    print_error("clCreateContext()",err);
    exit(EXIT_FAILURE);
  }

  /* create command queue */
  commands = clCreateCommandQueue(context,device_id,0,&err);
  if(err){
    print_error("clCreateCommandQueue()",err);
    exit(EXIT_FAILURE);
  }

  /* build program */
  program = clCreateProgramWithSource(context,1,(const char **) &kernel_source,NULL,&err);
  err = clBuildProgram(program,0,NULL,NULL,NULL,NULL);
  if(err){
    size_t len;
    char buffer[2048];
    clGetProgramBuildInfo(program,device_id,CL_PROGRAM_BUILD_LOG,sizeof(buffer),buffer,&len);
    printf("%s\n",buffer);
    exit(EXIT_FAILURE);
  }

  free_cl_source(kernel_source);

  /* setup filters and result on host */
  const size_t filter_width = atoi(argv[4]);
  const size_t filter_len = filter_width*filter_width;
  const unsigned int num_filters = atoi(argv[3]);
  const size_t image_size = image.width*image.height;
  float *h_filter = malloc(sizeof(float)*filter_len*num_filters);

  /* get a Gaussian */
  filter_Gauss2dbank(h_filter,num_filters,filter_width);

  uint8_t *h_result = malloc(sizeof(uint8_t)*image_size*num_filters);
  memset(h_result,0,sizeof(uint8_t)*image_size*num_filters);

  /* set up device memory and load image and filter data */
  d_image = clCreateBuffer(context,CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,sizeof(uint8_t)*image_size,image.bits,NULL);
  d_filter = clCreateBuffer(context,CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,sizeof(float)*filter_len*num_filters,h_filter,NULL);
  d_result = clCreateBuffer(context,CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,sizeof(uint8_t)*image_size*num_filters,h_result,NULL);

  kernel = clCreateKernel(program,"convolve2d",&err);
  if(err){
    print_error("clCreateKernel()",err);
    exit(EXIT_FAILURE);
  }

  /* send kernel arguments */
  err = clSetKernelArg(kernel,0,sizeof(cl_mem),&d_image);
  err |= clSetKernelArg(kernel,1,sizeof(cl_mem),&d_filter);
  err |= clSetKernelArg(kernel,2,sizeof(cl_mem),&d_result);
  err |= clSetKernelArg(kernel,3,sizeof(unsigned int),&image.width);
  err |= clSetKernelArg(kernel,4,sizeof(unsigned int),&image.height);
  err |= clSetKernelArg(kernel,5,sizeof(size_t),&filter_width);
  err |= clSetKernelArg(kernel,6,sizeof(size_t),&filter_width);
  err |= clSetKernelArg(kernel,7,sizeof(unsigned int),&num_filters);

  /* write buffers to global memory */
  err = clEnqueueWriteBuffer(commands,d_image,CL_FALSE,0,sizeof(uint8_t)*image_size,image.bits,0,NULL,NULL);
  err = clEnqueueWriteBuffer(commands,d_filter,CL_FALSE,0,sizeof(float)*filter_len*num_filters,h_filter,0,NULL,NULL);

  const size_t global[2] = {image_size, num_filters};

  /* enqueue kernel for execution */
  err = clEnqueueNDRangeKernel(commands,kernel,2,NULL,global,NULL,0,NULL,NULL);

  /* read from buffer after all commands have finished */
  err = clEnqueueReadBuffer(commands,d_result,CL_TRUE,0,sizeof(uint8_t)*image_size*num_filters,h_result,0,NULL,NULL);

  /* put result into image */
  memcpy(image.bits,h_result,sizeof(uint8_t)*image_size);
  /* save output */
  FreeImage_Save(FIF_JPEG,image.bitmap,"gray.jpg",JPEG_DEFAULT);

  free(h_filter);
  free(h_result);
  free(platform_ids);
  clReleaseKernel(kernel);
  clReleaseMemObject(d_image);
  clReleaseMemObject(d_filter);
  clReleaseMemObject(d_result);
  clReleaseProgram(program);
  clReleaseCommandQueue(commands);
  clReleaseContext(context);
  gimc_image_unload(&image);
  return 0;
}
