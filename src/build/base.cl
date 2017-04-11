/* convolves an image with many filters
 * for simplicity all filters have the same size and are square
 * image and result are assumed to be grayscale with a depth of 8 bits.
 */
__kernel
void convolve2d(__global unsigned char *image,
  __global float *filter,
  __global unsigned char *result,
  unsigned int image_width,
  unsigned int image_height,
  unsigned int filter_width,
  unsigned int filter_height,
  unsigned int num_filters)
{
  int pixel = get_global_id(0); /* current pixel */
  int fid = get_global_id(1); /* index of filter */

  if(pixel < (image_width*image_height) && fid < num_filters){

    const int px = pixel % image_width;
    const int py = ((pixel - px)/image_width);
    const unsigned int image_size = image_width * image_height;
    const unsigned int filter_len = filter_width * filter_height;
    const int offset = (filter_len - 1)/2;
    /* top left corner of filter window on image */
    const int cornerx = px - offset;
    const int cornery = py - offset;

    float sum = 0;
    /* iterate over the filter */
    for(unsigned int i = 0; i < filter_len; ++i){
      int col = (cornerx) + (i % filter_width);
      int row = (cornery) + ((i -(i%filter_width))/filter_width);

      float source;

      /* for row and column, if either go out of bounds, wrap the pixels */
      if(col < 0 || col > image_width - offset){
        col = col % (image_width - offset);
      }
      if(row < 0 || row > image_height-offset){
        row = row % (image_height - offset);
      }

      source = image[row*image_width + col];

      /* convolution uses the filter backwards */
      const unsigned int findex = filter_len - i - 1 + fid*filter_len;
      const float weight = filter[findex];
      sum += source*weight;
    }
    result[py*image_width + px] = sum;
  }
}
