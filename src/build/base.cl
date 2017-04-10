/* convolves an image with many filters
 * for simplicity all filters have the same size
 * image and result are assumed to be grayscale with a depth of 8 bits.
 */
__kernel
void convolve2d(__global unsigned char *image,
  __global float *filter,
  __global unsigned char *result,
  unsigned int image_width,
  unsigned int image_height,
  unsigned int filter_len,
  unsigned int num_filters)
{
  int pixel = get_global_id(0); /* current pixel */
  int fid = get_global_id(1); /* index of filter */
  printf("%d %d\n",pixel,fid);
}
